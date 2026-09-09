export const metrics = {
  intent: {
    accuracy: 96.2,
    delta: "+0.4%",
    f1_micro: 0.94,
    f1_macro: 0.92,
  },
  hallucination: {
    index: 0.08,
  },
  judge: {
    agreement: 98.6,
  },
  baselineComparison: {
    v4_2: {
      intent: 98.2,
      entity: 94.7,
      policy: 99.4,
    },
    v4_1: {
      intent: 95.1,
      entity: 92.9,
      policy: 98.8,
    }
  },
  failureBreakdown: [
    { name: "Policy Guardrail Trip", percentage: 2.1, color: "bg-secondary-container" },
    { name: "Context Window Overflow", percentage: 1.2, color: "bg-primary-fixed-dim" },
    { name: "Semantic Drift", percentage: 0.7, color: "bg-outline" },
  ],
  crossDomain: [
    { name: "Fintech", score: 99.1, color: "text-primary" },
    { name: "Health", score: 97.8, color: "text-primary" },
    { name: "SaaS", score: 99.5, color: "text-secondary-fixed" },
    { name: "Retail", score: 98.4, color: "text-primary" },
  ]
};

export const messages = [
  {
    index: 1,
    id: "TKT-89421",
    user: "@dev_kyle",
    avatar: "DK",
    tweet: "@AmazonHelp Can you check the status of my order? Tracking says 'out for delivery' but it's been that way for 3 days.",
    intent: "order_status_inquiry",
    confidence: 90,
    escalation: {
      decision: "auto_handle",
      reason: "Intent classified as 'order_status_inquiry' with 90% confidence. No high-urgency signals detected. Retrieved grounding examples are available to produce an accurate, on-topic automated reply.",
    },
    retrieval: [
      {
        id: "KB-882",
        title: "Order Status Inquiry",
        similarity: 0.84,
        snippet: "@AmazonHelp The package never arrived :( past 8pm, tracking still says 'out for delivery'"
      },
      {
        id: "Runbook-14",
        title: "Stale Tracking",
        similarity: 0.77,
        snippet: "12 days since placing my order and no delivery, meant to be delivered last Friday..."
      }
    ],
    draft: "@User I’m sorry you’ve been seeing “out for delivery” for three days—that’s definitely frustrating. Could you please share the carrier name and your tracking number (or order ID) so I can investigate further? You can reply here or send us a DM. ^AB",
    meta: {
      tier: "Enterprise",
      seats: "4.2k",
      mrr: "$8,400"
    }
  },
  {
    index: 3,
    id: "TKT-89422",
    user: "@nexus_corp",
    avatar: "NC",
    tweet: "@AmazonHelp someone hacked my account and made purchases I didn't authorise. Please help immediately, this is urgent.",
    intent: "account_access",
    confidence: 96,
    escalation: {
      decision: "escalate",
      reason: "Customer reports a potential account security breach (hacked) classified as account_access. This requires immediate action through a secure, authenticated channel — not an automated public reply.",
    },
    retrieval: [
      {
        id: "KB-102",
        title: "Account Hacked",
        similarity: 0.83,
        snippet: "someone hacked my account like a month ago I was suppose to get an email..."
      }
    ],
    draft: "@User I’m so sorry to hear that your account was compromised and unauthorized purchases were made. Let’s get this sorted right away: please visit https://www.amazon.com/gp/help/contact-us, select ‘Account & Login’ → ‘Account security’, and follow the steps to secure your account…",
    meta: {
      tier: "Pro",
      seats: "150",
      mrr: "$1,200"
    }
  },
  {
    index: 4,
    id: "TKT-89423",
    user: "@sarah_m",
    avatar: "SM",
    tweet: "@AmazonHelp I paid for Prime 2-day shipping and my package arrived 6 days late. This is the second time this month.",
    intent: "shipping_delay_complaint",
    confidence: 95,
    escalation: {
      decision: "auto_handle",
      reason: "Intent classified as 'shipping_delay_complaint' with 95% confidence. No high-urgency signals detected.",
    },
    retrieval: [
      {
        id: "KB-991",
        title: "Prime Delivery Delay",
        similarity: 0.86,
        snippet: "October 31st? Um that’s not 2 day shipping. That’s 4 day shipping. I bought Prime to avoid this..."
      }
    ],
    draft: "@User I’m really sorry your Prime shipment arrived six days late, especially after this happening twice this month. I’ve located your recent order and will issue a full refund",
    meta: {
      tier: "Basic",
      seats: "1",
      mrr: "$15"
    }
  },
  {
    index: 7,
    id: "TKT-89424",
    user: "@marcus_t",
    avatar: "MT",
    tweet: "@AmazonHelp This is absolutely disgusting. Worst customer service ever. Incompetent, pathetic, appalling service. I am furious and done with Amazon.",
    intent: "general_complaint",
    confidence: 90,
    escalation: {
      decision: "escalate",
      reason: "High-negativity language density detected (6 signals: disgusting, Incompetent, pathetic). An automated reply is unlikely to de-escalate a customer expressing this level of frustration.",
    },
    retrieval: [
      {
        id: "KB-11",
        title: "General Complaint",
        similarity: 0.87,
        snippet: "I can't believe this crap from you're customer service it's beyond a joke. This is unacceptable."
      }
    ],
    draft: "@UserHandle I’m really sorry you’ve had such a frustrating experience. To help fix this, could you please DM us your order number or a brief description of what happened so we can investigate and make things right?",
    meta: {
      tier: "Enterprise",
      seats: "8.1k",
      mrr: "$16,200"
    }
  }
];
