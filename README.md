# AI-Powered refugee Document Retrieval and Verification System

AI-Powered Refugee Document Retrieval and Verification System
Project Overview
The AI-Powered Document Retrieval and Verification System is designed to assist refugees in recovering and verifying lost academic documents from their home countries. By utilizing advanced AI and blockchain technology, the system provides a secure, efficient, and user-friendly solution for document retrieval and validation. The project is initially focused on refugees from DR Congo and South Sudan, specifically targeting those present in the Nakivale refugee settlement.

Key Components and Data Integration
Document Request Generation (NLP Model)

Data:
Text data of formal document request letters.
Database of educational institutions in DR Congo and South Sudan.
Parallel corpora for language translation.
Model: Fine-tuned NLP model for generating and translating request letters.
Integration: The model automatically generates and translates formal requests based on user inputs.
Document Tracking (Predictive Model)

Data:
Historical document request data.
Communication logs between users, institutions, and the platform.
Model: Predictive model to estimate document retrieval status and time.
Integration: Provides real-time status updates and estimated completion times.
Document Verification (AI-Based Validation Model)

Data:
Scanned copies of legitimate documents.
Associated metadata (issuing institution, date, type).
Blockchain records for verification.
Model: Document verification model using CNN for image recognition and OCR for text extraction.
Integration: Validates document authenticity by cross-checking with existing records and blockchain logs.
AI Chatbot Support (Conversational AI Model)

Data:
FAQ data and common issues.
Historical support tickets.
Model: Conversational AI model trained on Q&A data.
Integration: Powers a chatbot to provide real-time support and guidance during the document retrieval process.
Training Process and AI Integration
Data Collection: Gather data relevant to each component, focusing on the educational institutions and document types specific to DR Congo and South Sudan.
Model Training: Train each model on its respective data, ensuring accuracy and relevance to the specific needs of the target populations.
Integration: Deploy the models within the system to automate the process from request generation to document verification and user support.
Streamlit App Functionality
User Interface Layout

Homepage: Brief introduction and navigation options such as "Request Document," "Track Request," "Verify Document," and "Support."
Request Document Page: Allows users to fill in personal information and document details, then submit the request.
Track Request Page: Enables users to track the status of their document request.
Verify Document Page: Provides an option to upload and verify documents.
Support Page: Includes an AI-powered chatbot and a contact form for additional assistance.
Workflow

User Interaction: Users input their information and submit a document request.
AI-Generated Request: The NLP model generates a formal request in the selected language.
Request Tracking: The predictive model provides updates on the status and expected retrieval time.
Document Upload & Verification: Users upload the retrieved document, which is then verified using AI and blockchain.
Support: The AI chatbot offers guidance and troubleshooting throughout the process.
Next Steps
As the project develops, the scope may expand to include additional countries and refugee populations. The initial focus remains on ensuring the system effectively serves the specific needs of refugees from DR Congo and South Sudan in the Nakivale refugee settlement.

