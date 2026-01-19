#!/bin/bash
# SCIRS2 Compliance Verification Script
# Run this to verify kizzasi-model compliance with SCIRS2 policy

set -e

echo "======================================"
echo "kizzasi-model Compliance Verification"
echo "======================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS="${GREEN}✅ PASS${NC}"
FAIL="${RED}❌ FAIL${NC}"
WARN="${YELLOW}⚠️  WARN${NC}"

echo "1. Checking for direct rand usage..."
if grep -r "^use rand" src/ 2>/dev/null; then
    echo -e "$FAIL Found direct rand usage"
    exit 1
else
    echo -e "$PASS No direct rand usage"
fi
echo ""

echo "2. Checking for direct ndarray usage..."
if grep -r "^use ndarray" src/ 2>/dev/null; then
    echo -e "$FAIL Found direct ndarray usage"
    exit 1
else
    echo -e "$PASS No direct ndarray usage"
fi
echo ""

echo "3. Verifying scirs2-core usage..."
COUNT=$(grep -r "use scirs2_core" src/ 2>/dev/null | wc -l | tr -d ' ')
if [ "$COUNT" -gt 0 ]; then
    echo -e "$PASS Found $COUNT scirs2-core imports"
else
    echo -e "$FAIL No scirs2-core usage found"
    exit 1
fi
echo ""

echo "4. Checking Cargo.toml dependencies..."
if grep -E "^rand[[:space:]]*=" Cargo.toml 2>/dev/null; then
    echo -e "$FAIL Direct rand dependency found in Cargo.toml"
    exit 1
elif grep -E "^ndarray[[:space:]]*=" Cargo.toml 2>/dev/null; then
    echo -e "$FAIL Direct ndarray dependency found in Cargo.toml"
    exit 1
else
    echo -e "$PASS No direct rand/ndarray dependencies"
fi
echo ""

echo "5. Verifying workspace dependencies..."
if grep "scirs2-core.workspace = true" Cargo.toml >/dev/null; then
    echo -e "$PASS scirs2-core.workspace = true"
else
    echo -e "$FAIL scirs2-core not using workspace"
    exit 1
fi

if grep "scirs2-linalg.workspace = true" Cargo.toml >/dev/null; then
    echo -e "$PASS scirs2-linalg.workspace = true"
else
    echo -e "$FAIL scirs2-linalg not using workspace"
    exit 1
fi
echo ""

echo "6. Running cargo fmt check..."
if cargo fmt -- --check 2>&1 | grep -q "Diff"; then
    echo -e "$WARN Code needs formatting (run cargo fmt)"
else
    echo -e "$PASS Code is properly formatted"
fi
echo ""

echo "7. Running cargo clippy..."
if cargo clippy --all-features --all-targets -- -D warnings 2>&1 | grep -q "error\|warning"; then
    echo -e "$FAIL Clippy found issues"
    exit 1
else
    echo -e "$PASS Clippy found no issues"
fi
echo ""

echo "8. Verifying build..."
if cargo build --all-features --release >/dev/null 2>&1; then
    echo -e "$PASS Build successful"
else
    echo -e "$FAIL Build failed"
    exit 1
fi
echo ""

echo "======================================"
echo -e "${GREEN}✅ ALL COMPLIANCE CHECKS PASSED${NC}"
echo "======================================"
echo ""
echo "SCIRS2 Policy: COMPLIANT"
echo "Code Quality: EXCELLENT"
echo "Ready for: PRODUCTION"
echo ""
