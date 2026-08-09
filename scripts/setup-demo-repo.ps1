<#
.SYNOPSIS
  Create a throwaway GitHub repo seeded with pull requests containing known,
  planted defects, so the review agent can be evaluated against a known answer key.

.DESCRIPTION
  Four PRs are opened, each testing a different behavior:

    1. introduces  - off-by-one + swallowed exception   (expect HIGH)
    2. fixes       - guards a real null dereference      (expect fixes[], no bugs)
    3. security    - SQL injection via string building   (expect CRITICAL)
    4. clean       - rename and docstrings only          (expect SILENCE)

  PR 4 is the important one. A demo where everything gets flagged proves nothing
  about precision; the agent must stay quiet on a behavior-preserving change.

  Safe to re-run: an existing empty repo is reused rather than recreated.

.EXAMPLE
  .\scripts\setup-demo-repo.ps1 -RepoName prime-review-demo
#>

[CmdletBinding()]
param(
    [string]$RepoName = "prime-agent-review-demo",
    [ValidateSet("private", "public")]
    [string]$Visibility = "private"
)

# Native commands write warnings to stderr (git's CRLF notice, for one). Under
# ErrorActionPreference=Stop those become terminating errors and abort the script,
# so exit codes are checked explicitly instead.
$ErrorActionPreference = "Continue"

function Invoke-Checked {
    param(
        [Parameter(Mandatory)][string]$What,
        [Parameter(Mandatory)][scriptblock]$Command
    )
    $output = & $Command 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed (exit $LASTEXITCODE):`n$($output -join "`n")"
    }
    return $output
}

function Assert-Prereqs {
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        throw "gh CLI not found on PATH. Open a new terminal, or install from https://cli.github.com/"
    }
    gh auth status *> $null
    if ($LASTEXITCODE -ne 0) { throw "gh is not authenticated. Run: gh auth login" }
}

function New-DemoBranch {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][scriptblock]$Change,
        [Parameter(Mandatory)][string]$Title,
        [Parameter(Mandatory)][string]$Body
    )
    Invoke-Checked "git checkout main" { git checkout -q main }
    Invoke-Checked "git checkout -b $Name" { git checkout -q -b $Name }
    & $Change
    Invoke-Checked "git add" { git add -A }
    Invoke-Checked "git commit" { git commit -q -m $Title }
    Invoke-Checked "git push $Name" { git push -q -u --force origin $Name }
    $existing = gh pr list --head $Name --json number --jq '.[0].number' 2>$null
    if ($LASTEXITCODE -eq 0 -and $existing) {
        Write-Host "  PR already open for $Name (#$existing)" -ForegroundColor Yellow
        return
    }
    Invoke-Checked "gh pr create" { gh pr create --title $Title --body $Body --base main --head $Name }
    Write-Host "  opened PR: $Title" -ForegroundColor Green
}

Assert-Prereqs

$owner = (Invoke-Checked "gh api user" { gh api user --jq .login }).Trim()
$slug = "$owner/$RepoName"
$workdir = Join-Path $env:TEMP "prime-demo-$(Get-Random)"
New-Item -ItemType Directory -Force -Path $workdir | Out-Null
Push-Location $workdir

try {
    gh repo view $slug *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Reusing existing repo $slug" -ForegroundColor Yellow
    }
    else {
        Write-Host "Creating $slug ..." -ForegroundColor Cyan
        Invoke-Checked "gh repo create" {
            gh repo create $RepoName --$Visibility --description "Throwaway repo for evaluating the Prime PR review agent"
        }
    }

    Invoke-Checked "git init" { git init -q -b main }
    # Suppress the CRLF conversion notice; it is noise here and it writes to stderr.
    Invoke-Checked "git config autocrlf" { git config core.autocrlf false }
    Invoke-Checked "git config safecrlf" { git config core.safecrlf false }
    Invoke-Checked "git remote add" { git remote add origin "https://github.com/$slug.git" }

    # ---------- baseline ----------
    New-Item -ItemType Directory -Force -Path "shop" | Out-Null

    @'
"""Order processing."""


def total_price(items):
    """Sum price * quantity across every item."""
    return sum(item["price"] * item["qty"] for item in items)


def apply_discount(total, percent):
    """Reduce a total by a percentage."""
    return total * (1 - percent / 100)


def find_order(orders, order_id):
    """Return the order with this id, or None."""
    for order in orders:
        if order["id"] == order_id:
            return order
    return None


def order_summary(orders, order_id):
    """Human-readable summary for one order."""
    order = find_order(orders, order_id)
    return f"Order {order['id']}: {total_price(order['items']):.2f}"
'@ | Set-Content "shop/orders.py" -Encoding utf8

    @'
"""Customer lookups."""
import sqlite3


def get_customer(conn, customer_id):
    """Fetch one customer by id."""
    cursor = conn.execute(
        "SELECT id, name, email FROM customers WHERE id = ?", (customer_id,)
    )
    return cursor.fetchone()
'@ | Set-Content "shop/customers.py" -Encoding utf8

    @'
# Shop

Deliberately small demo service. Used to evaluate an automated PR reviewer.
'@ | Set-Content "README.md" -Encoding utf8

    Invoke-Checked "git add baseline" { git add -A }
    Invoke-Checked "git commit baseline" { git commit -q -m "Initial commit: order and customer modules" }
    # Force-push: this repo is throwaway scaffolding, and a partially-completed
    # earlier run may have left commits on the remote.
    Invoke-Checked "git push main" { git push -q -u --force origin main }
    Write-Host "  baseline pushed" -ForegroundColor Green

    # ---------- PR 1: introduces bugs ----------
    New-DemoBranch -Name "perf/skip-last-item" -Title "Speed up total_price" `
        -Body "Avoids iterating the whole list." -Change {
        @'
"""Order processing."""


def total_price(items):
    """Sum price * quantity across every item."""
    return sum(item["price"] * item["qty"] for item in items[:-1])


def apply_discount(total, percent):
    """Reduce a total by a percentage."""
    try:
        return total * (1 - percent / 100)
    except Exception:
        pass


def find_order(orders, order_id):
    """Return the order with this id, or None."""
    for order in orders:
        if order["id"] == order_id:
            return order
    return None


def order_summary(orders, order_id):
    """Human-readable summary for one order."""
    order = find_order(orders, order_id)
    return f"Order {order['id']}: {total_price(order['items']):.2f}"
'@ | Set-Content "shop/orders.py" -Encoding utf8
    }

    # ---------- PR 2: fixes a real bug ----------
    New-DemoBranch -Name "fix/missing-order-guard" -Title "Guard against a missing order" `
        -Body "order_summary crashed when find_order returned None." -Change {
        $text = Get-Content "shop/orders.py" -Raw
        $text = $text.Replace(
@'
    order = find_order(orders, order_id)
    return f"Order {order['id']}: {total_price(order['items']):.2f}"
'@,
@'
    order = find_order(orders, order_id)
    if order is None:
        return f"Order {order_id}: not found"
    return f"Order {order['id']}: {total_price(order['items']):.2f}"
'@)
        $text | Set-Content "shop/orders.py" -Encoding utf8
    }

    # ---------- PR 3: security ----------
    New-DemoBranch -Name "feat/customer-search" -Title "Add customer search by name" `
        -Body "Supports partial name matching." -Change {
        @'
"""Customer lookups."""
import sqlite3


def get_customer(conn, customer_id):
    """Fetch one customer by id."""
    cursor = conn.execute(
        "SELECT id, name, email FROM customers WHERE id = ?", (customer_id,)
    )
    return cursor.fetchone()


def search_customers(conn, name):
    """Find customers whose name matches a search term."""
    query = f"SELECT id, name, email FROM customers WHERE name LIKE '%{name}%'"
    return conn.execute(query).fetchall()
'@ | Set-Content "shop/customers.py" -Encoding utf8
    }

    # ---------- PR 4: clean, must stay silent ----------
    New-DemoBranch -Name "chore/clarify-naming" -Title "Clarify parameter naming in apply_discount" `
        -Body "Renames a parameter and expands a docstring. No behavior change." -Change {
        $text = Get-Content "shop/orders.py" -Raw
        $text = $text.Replace(
@'
def apply_discount(total, percent):
    """Reduce a total by a percentage."""
    return total * (1 - percent / 100)
'@,
@'
def apply_discount(total, discount_percent):
    """Reduce a total by a percentage.

    Args:
        total: The pre-discount amount.
        discount_percent: Percentage to remove, 0-100.
    """
    return total * (1 - discount_percent / 100)
'@)
        $text | Set-Content "shop/orders.py" -Encoding utf8
    }

    Invoke-Checked "git checkout main" { git checkout -q main }

    Write-Host ""
    Write-Host "Done. Repo: https://github.com/$slug" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Answer key:" -ForegroundColor Yellow
    Write-Host "  PR 'Speed up total_price'          -> HIGH: off-by-one; swallowed exception returns None"
    Write-Host "  PR 'Guard against a missing order' -> fixes[]: null dereference. No new bugs."
    Write-Host "  PR 'Add customer search by name'   -> CRITICAL: SQL injection"
    Write-Host "  PR 'Clarify parameter naming'      -> SILENT. Must not post."
    Write-Host ""
    Write-Host "Now set in config.toml:" -ForegroundColor Yellow
    Write-Host "  [repo]   owner = `"$owner`"   name = `"$RepoName`""
    Write-Host "  [review] bot_login = `"`"   (empty, so your own PRs are still reviewed)"
}
finally {
    Pop-Location
}
