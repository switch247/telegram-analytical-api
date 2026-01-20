Interim Report: Intelligent Complaint Analysis for CrediTrust Financial
Project Title: Building a RAG-Powered Chatbot for Strategic Feedback Insights
Author: Abel
Date: January 4, 2026
Executive Summary
This project aims to modernize CrediTrust Financial's complaint analysis by building a RAG-powered AI tool. We have completed the foundational data engineering phase, which includes cleaning the CFPB dataset and building a high-performance vector search engine using ChromaDB. This foundation allows internal stakeholders to move from manual reading to automated, semantic insights.
1. Business Objective and Understanding
CrediTrust Financial is currently at a critical junction in its expansion. As a mobile-first digital finance provider serving over 500,000 users across East Africa, the company’s success depends on its ability to respond to customer needs across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.
However, rapid growth has resulted in a high volume of unstructured customer complaints. Currently, the process is reactive and manual:
The Bottleneck: Stakeholders like Asha (Product Manager) spend hours manually reading individual narratives. With thousands of complaints arriving monthly, identifying a single systemic issue can take days of manual collation.
The Technical Gap: Non-technical teams like Support and Compliance are dependent on data analysts to generate simple reports. This creates a lag in response time that can impact customer retention and regulatory standing.
The Strategic Objective: Our mission is to build an internal AI tool that acts as a "strategic brain." This RAG-powered system will allow Asha to ask direct questions—such as "What are the primary reasons for card activation failures in the last 30 days?"—and receive synthesized, evidence-backed answers in seconds.
KPIs for Success:
Speed to Insight: Reduce trend identification latency from 48-72 hours to under 2 minutes.
Autonomy: 100% self-service capability for non-technical managers for basic trend analysis.
Proactive Strategy: Integration of complaint themes into the bi-weekly product development sprint cycle.
2. Task 1: Complaints Analysis, EDA & Preprocessing
The primary goal of Task 1 was to distill a massive dataset into a high-quality corpus for the RAG system.
2.1 Categorical Observations & Market Volume
Our analysis focused on isolating CrediTrust's four distinct entities. We observed that Credit Cards and Personal Loans represent over 70% of the total volume, while Savings Accounts and Money Transfers, though lower in volume, contain highly technical system-related feedback.
Supporting Evidence:
	
Figure 1: Distribution of Top 8 Products — Note: Shows Credit Card, Personal Loan, Savings Account, and Money Transfer as distinct, core categories.

Figure 2: Top 10 Product

Figure 3: Top 10 Submitted via
2.2 Narrative Depth and Temporal Trends
Narrative Volume: After removing records missing the Consumer complaint narrative field, we retained over 50,000 interactions.
Text Complexity: The average narrative length sits between 150 and 400 words. This provides sufficient signal for semantic extraction.
Supporting Evidence:
	
Figure 4: Distribution of Complaint Narrative Length (Word Count)

Figure 5: Monthly Complaints Trend
2.3 Text Cleaning Pipeline
We implemented a pipeline involving lowercasing, removal of non-alphanumeric characters, and Boilerplate Reduction, which programmatically removed high-frequency phrases like "I am writing to file a complaint" to increase the density of relevant keywords.
3. Task 2: Embedding and Vector Store Indexing
3.1 Stratified Sampling Strategy
To maintain a representative vector store, we implemented a stratified sampling strategy. This ensures that the vector store maintains the original proportion of our four core products, preventing the RAG system from becoming biased toward high-volume Credit Card issues at the expense of Savings Account insights.
3.2 Chunking and Vectorization
Using RecursiveCharacterTextSplitter, we segmented narratives with a chunk_size of 1000 and chunk_overlap of 200. We deployed the all-MiniLM-L6-v2 model to convert these into 384-dimensional vectors stored in ChromaDB.
4. Next Steps(Roadmap for Task 3 and Task 4)
4.1 Task 3: RAG Pipeline & Qualitative Evaluation
We will integrate the ChromaDB retriever with an LLM (GPT-4o) using LangChain. To ensure the system meets Asha's needs, we have defined the following 10 test questions for qualitative evaluation:
"What are the top 3 reasons customers are unhappy with Credit Card interest rates?"
"List recurring technical issues mentioned regarding Money Transfers."
"Summarize complaints related to the Personal Loan application process."
"Are there specific trends in how Savings Account fees are being reported?"
"What evidence is there of system downtime affecting card activations?"
"Identify the most frequent complaint regarding customer service response times."
"How do users describe their experience with the mobile app's transfer feature?"
"What are the common complaints about loan repayment schedules?"
"Synthesize customer feedback regarding hidden fees in Savings Accounts."
"Summarize regulatory concerns mentioned in the Money Transfer narratives."
4.2 Task 4: Interactive UI Development
The final phase involves building a Gradio interface. The UI will feature:
Text Input: A central query bar for natural language questions.
Submit Button: To trigger the RAG chain.
Answer Display: A formatted text area for the LLM's synthesized response.
Source Citation: A dedicated "Evidence" section showing the specific complaint IDs used to generate the answer.
Streaming Output: Real-time text generation to improve user experience (UX).
4.3 Potential Challenges
Hallucination Risk: Mitigated by strict "grounding" in the retrieved documents.
Latency: Optimizing the retrieval-to-generation loop to keep responses under 2 seconds.


