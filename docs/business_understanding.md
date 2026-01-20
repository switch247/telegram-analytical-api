# Business Understanding: Intelligent Complaint Analysis for Financial Services

## Business Objective
CrediTrust Financial is a fast-growing digital finance company serving East African markets through a mobile-first platform. Their offerings span across:

With a user base of over 500,000 and operations expanding into three countries, CrediTrust receives thousands of customer complaints per month through its in-app channels, email, and regulatory reporting portals.

## Business Understanding

This project aims to generate actionable insights about Ethiopian medical businesses by analyzing data scraped from public Telegram channels. The platform is designed for Kara Solutions, a data science consultancy, to help answer key business questions such as:

- What are the top 10 most frequently mentioned medical products or drugs across all channels?
- How does the price or availability of a specific product vary across different channels?
- Which channels have the most visual content (e.g., images of pills vs. creams)?
- What are the daily and weekly trends in posting volume for health-related topics?

The solution leverages a modern ELT pipeline, including a data lake, PostgreSQL data warehouse, dbt for transformation, YOLOv8 for image enrichment, and FastAPI for analytics.

### Data Sources
- Public Telegram channels focused on medical, pharmaceutical, and cosmetic products in Ethiopia.

### Stakeholders
- Kara Solutions (client)
- Data engineering and analytics teams

### Business Value
- Improved market intelligence for medical and pharmaceutical products
- Enhanced ability to track trends and product availability
- Visual content analysis for marketing and product display insights

For a detailed breakdown of tasks and deliverables, see `experiments/todo.md`.

## Key Performance Indicators (KPIs)
The success of the project will be measured against three KPIs:
1. **Efficiency**: Decrease the time it takes for a Product Manager to identify a major complaint trend from days to minutes.
2. **Accessibility**: Empower non-technical teams (like Support and Compliance) to get answers without needing a data analyst.
3. **Proactivity**: Shift the company from reacting to problems to proactively identifying and fixing them based on real-time customer feedback.

## Motivation
CrediTrust’s internal teams face serious bottlenecks:

## Proposed Solution
As a Data & AI Engineer, the task is to develop an intelligent complaint-answering chatbot that empowers product, support, and compliance teams to understand customer pain points across five major product categories:

The goal is to build a **Retrieval-Augmented Generation (RAG)** agent that:
1. Allows internal users to ask plain-English questions about customer complaints (e.g., “Why are people unhappy with Credit Cards?”).
2. Uses semantic search (via a vector database like FAISS or ChromaDB) to retrieve the most relevant complaint narratives.
3. Feeds the retrieved narratives into a language model (LLM) that generates concise, insightful answers.
4. Supports multi-product querying, making it possible to filter or compare issues across financial services.
