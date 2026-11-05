> FICTIONAL EXAMPLE. Brightloop Software GmbH and Brightloop HR Cloud do not exist. This file is demo input for the AI Act evidence pack.

# Brightloop HR Cloud

Brightloop HR Cloud is a SaaS applicant tracking system for mid-sized European companies. Customers use it in the browser and through a REST API.

## AI features

- **Candidate assistant.** A chat assistant on the careers page answers applicants' questions about open roles and the application process. The assistant opens every conversation with a notice that the applicant is talking to an AI assistant.
- **Job ad writer.** Recruiters type a few bullet points and the job ad writer drafts the full job advertisement text with AI. Recruiters edit the draft before publishing.
- **Shortlist ranking.** The system ranks incoming applications against the job requirements and suggests a shortlist to the recruiter. The final decision on every candidate is taken by a human recruiter.

## How it is built

The candidate assistant and the job ad writer call a third-party large language model (GPT-4o via the Azure OpenAI Service). Shortlist ranking uses a gradient-boosted model trained by Brightloop on anonymised historical hiring outcomes from consenting customers.

Every ranking request and response is written to an audit log that customers can export.

## Version

Current release: Brightloop HR Cloud 4.2.
