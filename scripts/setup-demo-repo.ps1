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

.EXAMPLE
  .\scripts\setup-demo-repo.ps1 -RepoName prime-review-demo
#>

[CmdletBinding()]
param(
    [string]$RepoName = "prime-agent-review-demo",
    [ValidateSet("private", "public")]
    [string]$Visibility = "private"
)

$ErrorActionPreference = "Stop"

function Assert-Prereqs {
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        throw "gh CLI not found on PATH. Open a new terminal, or install from https://cli.github.com/"
    }
    gh auth status *> $null
    if ($LASTEXITCODE -ne 0) { throw "gh is not authenticated. Run: gh auth login" }
}

function New-Branch {
    param([string]$Name, [scriptblock]$Change, [string]$Title, [string]$Body)

    git checkout -q main
    git checkout -q -b $Name
    & $Change
    git add -A
    git commit -q -m $Title
    git push -q -u origin $Name
    gh pr create --title $Title --body $Body --base main --head $Name | Out-Null
    Write-Host "  opened PR: $Title" -ForegroundColor Green
}

Assert-Prereqs

$owner = (gh api user --jq .login).Trim()
$workdir = Join-Path $env:TEMP "prime-demo-$(Get-Random)"
New-Item -ItemType Directory -Force -Path $workdir | Out-Null
Push-Location $workdir

try {
    Write-Host "Creating $owner/$RepoName ..." -ForegroundColor Cyan
    gh repo create $RepoName --$Visibility --description "Throwaway repo for evaluating the Prime PR review agent" | Out-Null

    git init -q -b main
    git remote add origin "https://github.com/$owner/$RepoName.git"

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

    git add -A
    git commit -q -m "Initial commit: order and customer modules"
    git push -q -u origin main
    Write-Host "  baseline pushed" -ForegroundColor Green

    # ---------- PR 1: introduces bugs ----------
    New-Branch -Name "perf/skip-last-item" -Title "Speed up total_price" -Body "Avoids iterating the whole list." -Change {
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
    New-Branch -Name "fix/missing-order-guard" -Title "Guard against a missing order" -Body "order_summary crashed when find_order returned None." -Change {
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
    New-Branch -Name "feat/customer-search" -Title "Add customer search by name" -Body "Supports partial name matching." -Change {
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
    New-Branch -Name "chore/clarify-naming" -Title "Clarify parameter naming in apply_discount" -Body "Renames a parameter and expands a docstring. No behavior change." -Change {
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

    git checkout -q main

    Write-Host ""
    Write-Host "Done. Repo: https://github.com/$owner/$RepoName" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Answer key:" -ForegroundColor Yellow
    Write-Host "  PR 1 'Speed up total_price'      -> HIGH: off-by-one; swallowed exception returning None"
    Write-Host "  PR 2 'Guard against missing order' -> fixes[]: null dereference. No new bugs."
    Write-Host "  PR 3 'Add customer search'       -> CRITICAL: SQL injection"
    Write-Host "  PR 4 'Clarify parameter naming'  -> SILENT. Must not post."
    Write-Host ""
    Write-Host "Now set in config.toml:" -ForegroundColor Yellow
    Write-Host "  [repo] owner = `"$owner`"   name = `"$RepoName`""
    Write-Host "  [review] bot_login = `"`"   (empty, so your own PRs are still reviewed)"
}
finally {
    Pop-Location
}
