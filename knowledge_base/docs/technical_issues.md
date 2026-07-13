# Technical Issues & Error Codes

## Common Error Codes

### Authentication Errors
| Code | Meaning | Resolution |
|---|---|---|
| E-1001 | Invalid credentials | Check email/password. Use "Forgot Password" if needed. |
| E-1002 | Session expired | Log out, clear browser cookies, log back in |
| E-1003 | Account locked | See Account Troubleshooting — submit ticket to unlock |
| E-1004 | 2FA code invalid | Ensure device clock is synced; request new code |
| E-1005 | Token expired | Re-initiate the action that generated the token |

### Application Errors
| Code | Meaning | Resolution |
|---|---|---|
| E-2001 | Internal server error | Wait 5 min and retry; check status page if persistent |
| E-2002 | Service temporarily unavailable | Check status.example.com for outage info |
| E-2003 | Request timeout | Slow connection or server load; retry in 1–2 minutes |
| E-3001 | File too large | Maximum upload size is 25 MB |
| E-3002 | Unsupported file type | Accepted types: PDF, PNG, JPG, CSV, XLSX |
| E-4001 | API rate limit exceeded | Basic: 1,000 calls/mo; Premium: 50,000 calls/mo |
| E-4023 | Login crash / initialization failure | Clear app cache, update to latest version, reinstall if needed |
| E-5001 | Payment processing error | Verify payment method; try a different card |
| E-5002 | Subscription not found | Contact support — account sync issue |

## Error E-4023 (Login Crash) — Detailed Troubleshooting

This is our most reported technical error. Follow these steps in order:

**Step 1: Clear app cache**
- Mobile: Settings → Apps → [App Name] → Clear Cache
- Browser: Ctrl+Shift+Delete → Clear Cached Images/Files

**Step 2: Force update**
- Mobile: App Store / Play Store → Check for updates
- Browser: Hard refresh (Ctrl+Shift+R)

**Step 3: Check OS compatibility**
- iOS 14+, Android 8+, macOS 11+, Windows 10+

**Step 4: Reinstall**
- Uninstall completely, restart device, reinstall fresh

**Step 5: Network check**
- Disable VPN if active
- Try on a different Wi-Fi network or cellular

**If all 5 steps fail:** Submit a ticket with:
- Your device model and OS version
- App version number
- Whether the crash happens immediately or after entering credentials
- Any additional error message text

> **Note:** If you've submitted multiple tickets about E-4023 that have not been resolved, your case will be escalated to our Tier-2 engineering team for a deep dive investigation.

## App Performance Issues

**Slow loading:**
- Clear browser cache / app cache
- Disable browser extensions
- Check internet speed (we recommend 5+ Mbps)
- Try incognito/private browsing mode

**Features not displaying correctly:**
- Hard refresh the page (Ctrl+Shift+R)
- Try a different browser
- Disable dark mode browser extensions

**Data not syncing:**
- Log out and log back in
- Check that you're not logged into two accounts simultaneously
- Wait up to 5 minutes for sync to propagate

## Reporting a Bug
We take bugs seriously. To report:
1. Submit a ticket with category "Bug Report"
2. Include: steps to reproduce, expected behavior, actual behavior, screenshots/video, browser/OS info
3. Our engineering team reviews bug reports within 48 hours
4. Confirmed bugs receive a tracking ID — we'll notify you when fixed
