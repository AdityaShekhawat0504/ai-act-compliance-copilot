> FICTIONAL EXAMPLE. Demo input only; describes a system that does not exist.

# Model card: Brightloop shortlist ranker v3

## Intended use

Ranks job applications against the requirements of a specific job posting to help recruiters decide whom to interview. The ranker scores each application from 0 to 100 and explains the top three factors behind the score.

## Training data

Trained on 410,000 anonymised applications and hiring outcomes from 37 customers who opted in to model training between 2022 and 2025. Names, photos, dates of birth and addresses are removed before training.

## Evaluation

We report precision@10 against recruiter shortlists and the selection-rate ratio between gender groups on a held-out test set. The selection-rate ratio must stay above 0.8 before a release ships.

## Limitations

The ranker performs worse for roles with fewer than 50 historical applications and for CVs that are not in English or German.

## Human oversight

Recruiters see the score and the top factors, can re-order the shortlist manually, and must confirm every rejection. No application is rejected automatically.
