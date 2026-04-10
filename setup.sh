#!/bin/bash
# Mnemosyne — Project Setup Script
# Run this in your mnemosyne project root after cloning

set -e

echo "=== Mnemosyne Agent Team Setup ==="

# 1. Create directory structure
echo "[1/5] Creating directory structure..."
mkdir -p src/{db/{models,repositories},pipeline/{extraction,embedding,episodes,consolidation,decay},rules,retrieval,context,providers,embedding,llm,integration/tools,config/defaults,monitoring}
mkdir -p rules/core
mkdir -p prompts
mkdir -p migrations
mkdir -p tests/{unit/{db,pipeline,rules,retrieval,context,providers,integration},integration/{db,e2e},benchmarks,diagnostics,fixtures}
mkdir -p sql
mkdir -p config
mkdir -p docs

# 2. Copy design doc
echo "[2/5] Setting up docs..."
if [ -f "agent-memory-design.md" ]; then
    cp agent-memory-design.md docs/agent-memory-design.md
fi

# 3. Install gstack (if not already installed)
echo "[3/5] Checking gstack..."
if [ ! -d "$HOME/.claude/skills/gstack" ]; then
    echo "  Installing gstack..."
    git clone --single-branch --depth 1 https://github.com/garrytan/gstack.git "$HOME/.claude/skills/gstack"
    "$HOME/.claude/skills/gstack/setup"
else
    echo "  gstack already installed."
fi

# 4. Create git worktrees for parallel agent work
echo "[4/5] Creating git worktrees..."
git init . 2>/dev/null || true
git add -A && git commit -m "Initial project structure" --allow-empty 2>/dev/null || true

for branch in database pipeline retrieval integration qa; do
    if [ ! -d "../mnemosyne-${branch}" ]; then
        git branch "feature/${branch}" 2>/dev/null || true
        git worktree add "../mnemosyne-${branch}" "feature/${branch}" 2>/dev/null || echo "  Worktree ${branch} may already exist"
    fi
done

# 5. Verify agent team config
echo "[5/5] Verifying agent configuration..."
if [ -f ".claude/settings.json" ]; then
    echo "  ✓ settings.json found with agent teams enabled"
fi

agent_count=$(ls .claude/agents/*.md 2>/dev/null | wc -l)
echo "  ✓ ${agent_count} agents configured:"
for f in .claude/agents/*.md; do
    echo "    - $(basename "$f" .md)"
done

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start working:"
echo "  1. cd into your project root"
echo "  2. Run: claude"
echo "  3. Tell Claude to create an agent team for your sprint"
echo ""
echo "Available agents:"
echo "  /agents    — view and manage agents in Claude Code"
echo ""
echo "Git worktrees created at:"
for branch in database pipeline retrieval integration qa; do
    echo "  ../mnemosyne-${branch} → feature/${branch}"
done
