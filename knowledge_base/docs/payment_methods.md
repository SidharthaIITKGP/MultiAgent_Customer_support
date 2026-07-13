# Payment Methods

## Accepted Payment Methods

### Credit & Debit Cards
- Visa (credit and debit)
- Mastercard (credit and debit)
- American Express
- Discover
- Cards must be issued in a supported currency; 3D Secure authentication may be required

### Digital Wallets
- PayPal (personal and business accounts)
- Apple Pay (available on Safari and iOS app)
- Google Pay (available on Chrome and Android app)

### Bank Transfer / ACH (US only)
- Available for annual subscriptions and orders over $200
- Processing time: 3–5 business days
- Set up in Account Settings → Billing → Add Bank Account

### Cryptocurrency (Beta)
- Bitcoin and Ethereum accepted for annual plan payments only
- Exchange rate locked at time of invoice generation
- Contact billing@example.com to request a crypto invoice

### Not Accepted
- Cash, money orders, checks
- Prepaid gift cards (technical limitation)
- Cards with billing addresses in sanctioned countries

## Currency Support
- Default: USD
- Available: EUR, GBP, CAD, AUD, JPY, INR, BRL (18 currencies total)
- Currency set by billing address at account creation
- Currency changes require a support ticket (affects future invoices only)

## Failed Payments

### Why Payments Fail
1. Insufficient funds
2. Card expired
3. Card blocked by issuing bank (international transaction flag)
4. Incorrect billing address / CVV
5. Card issuer flagged as suspicious activity

### What to Do When a Payment Fails
1. **Update your card**: Account Settings → Billing → Payment Methods
2. **Contact your bank**: Ask them to allow the charge from "Example Inc."
3. **Try a different card**: Add a backup payment method
4. **Use PayPal**: Often more permissive for international transactions

### Automatic Retry Schedule
- Day 0: Initial payment attempt fails → email notification
- Day 3: Automatic retry
- Day 7: Second automatic retry
- Day 10: Third and final retry; account suspended if still failing

## Security

### Is my payment information stored securely?
Yes. We use Stripe as our payment processor. Your full card number is never stored on our servers — only a tokenized reference. We are PCI-DSS Level 1 compliant.

### Fraud Protection
- Unusual payment activity triggers an automatic hold and email alert
- You can review and approve/reject flagged charges in Account Settings → Billing → Alerts
- Unauthorized charges: contact us immediately AND your card issuer

## Receipts and Invoices
- Automatic receipt sent to your billing email after every charge
- Download past invoices: Account Settings → Billing → Invoice History
- For business customers: VAT/GST invoice available in Account Settings → Billing → Tax Settings
