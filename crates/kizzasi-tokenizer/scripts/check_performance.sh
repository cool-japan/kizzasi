#!/usr/bin/env bash
# Performance regression detection script for kizzasi-tokenizer
#
# Usage:
#   ./scripts/check_performance.sh [base_branch] [threshold_percent]
#
# Arguments:
#   base_branch:        Branch to compare against (default: master)
#   threshold_percent:  Performance regression threshold in % (default: 10)
#
# Examples:
#   ./scripts/check_performance.sh master 10
#   ./scripts/check_performance.sh main 5

set -euo pipefail

# Configuration
BASE_BRANCH="${1:-master}"
THRESHOLD="${2:-10}"
WORK_DIR=$(mktemp -d)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Performance Regression Detection"
echo "=========================================="
echo "Base branch:    $BASE_BRANCH"
echo "Threshold:      $THRESHOLD%"
echo "Working dir:    $WORK_DIR"
echo ""

cleanup() {
    echo "Cleaning up..."
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

# Save current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Check if base branch exists
if ! git rev-parse --verify "$BASE_BRANCH" > /dev/null 2>&1; then
    echo -e "${RED}Error: Base branch '$BASE_BRANCH' does not exist${NC}"
    exit 1
fi

# Install required tools if not present
if ! command -v cargo-criterion &> /dev/null; then
    echo "Installing cargo-criterion..."
    cargo install cargo-criterion
fi

if ! command -v critcmp &> /dev/null; then
    echo "Installing critcmp..."
    cargo install critcmp
fi

echo ""
echo "=========================================="
echo "Step 1: Running benchmarks on current branch"
echo "=========================================="

# Run benchmarks on current code
cd "$PROJECT_ROOT"
cargo bench --all-features --bench comprehensive_benchmarks -- --save-baseline current

echo ""
echo "=========================================="
echo "Step 2: Switching to base branch"
echo "=========================================="

# Stash any uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Stashing uncommitted changes..."
    git stash push -m "perf-check-stash-$(date +%s)"
    STASHED=1
else
    STASHED=0
fi

# Switch to base branch
git checkout "$BASE_BRANCH"

echo ""
echo "=========================================="
echo "Step 3: Running benchmarks on base branch"
echo "=========================================="

# Run benchmarks on base code
cargo bench --all-features --bench comprehensive_benchmarks -- --save-baseline base

echo ""
echo "=========================================="
echo "Step 4: Switching back to current branch"
echo "=========================================="

# Switch back to original branch
git checkout "$CURRENT_BRANCH"

# Restore stashed changes if any
if [ "$STASHED" -eq 1 ]; then
    echo "Restoring stashed changes..."
    git stash pop
fi

echo ""
echo "=========================================="
echo "Step 5: Analyzing results"
echo "=========================================="

# Compare benchmarks
COMPARISON_FILE="$WORK_DIR/comparison.txt"
critcmp base current > "$COMPARISON_FILE" || true

# Display results
cat "$COMPARISON_FILE"
echo ""

# Parse results and check for regressions
REGRESSIONS=0
IMPROVEMENTS=0
TOTAL_BENCHMARKS=0

while IFS= read -r line; do
    # Look for percentage changes in the format: "x.xx% faster" or "x.xx% slower"
    if [[ $line =~ ([0-9]+\.[0-9]+)%[[:space:]]+(faster|slower) ]]; then
        PERCENT="${BASH_REMATCH[1]}"
        DIRECTION="${BASH_REMATCH[2]}"
        TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))

        if [ "$DIRECTION" = "slower" ]; then
            # Check if regression exceeds threshold
            if (( $(echo "$PERCENT > $THRESHOLD" | bc -l) )); then
                echo -e "${RED}⚠️  Regression detected: $PERCENT% slower${NC}"
                REGRESSIONS=$((REGRESSIONS + 1))
            fi
        elif [ "$DIRECTION" = "faster" ]; then
            IMPROVEMENTS=$((IMPROVEMENTS + 1))
        fi
    fi
done < "$COMPARISON_FILE"

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total benchmarks analyzed: $TOTAL_BENCHMARKS"
echo -e "${GREEN}Improvements:${NC} $IMPROVEMENTS"
echo -e "${RED}Regressions (>$THRESHOLD%):${NC} $REGRESSIONS"

# Save detailed report
REPORT_FILE="$PROJECT_ROOT/performance_report.txt"
{
    echo "Performance Regression Report"
    echo "============================="
    echo "Date: $(date)"
    echo "Base branch: $BASE_BRANCH"
    echo "Current branch: $CURRENT_BRANCH"
    echo "Threshold: $THRESHOLD%"
    echo ""
    echo "Results:"
    echo "--------"
    cat "$COMPARISON_FILE"
    echo ""
    echo "Summary:"
    echo "--------"
    echo "Total benchmarks: $TOTAL_BENCHMARKS"
    echo "Improvements: $IMPROVEMENTS"
    echo "Regressions (>$THRESHOLD%): $REGRESSIONS"
} > "$REPORT_FILE"

echo ""
echo "Detailed report saved to: $REPORT_FILE"

# Exit with error if regressions detected
if [ "$REGRESSIONS" -gt 0 ]; then
    echo ""
    echo -e "${RED}❌ Performance regressions detected!${NC}"
    exit 1
else
    echo ""
    echo -e "${GREEN}✅ No significant performance regressions detected${NC}"
    exit 0
fi
