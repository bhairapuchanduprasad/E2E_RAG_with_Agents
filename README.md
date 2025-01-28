# **Generative AI for Internal HR Q&A**

## 📌 **Problem Statement**
Covestro's HR department receives numerous queries from employees about policies, procedures, and benefits. Manually addressing these queries is inefficient and time-consuming, particularly for repetitive questions. 

### **Challenges:**
- Inefficient manual responses.
- Lack of a structured retrieval mechanism.
- Need for a scalable, privacy-focused, and automated solution.

## 🎯 **Proposed Solution**
This project implements **Retrieval-Augmented Generation (RAG) with an Agentic Workflow**, ensuring:
- **Efficient Retrieval**: Fetching precise HR-related documents.
- **Enhanced Accuracy**: Reducing hallucinations in LLM responses.
- **Scalability & Automation**: Using AWS-based architecture for seamless scaling.
- **Data Privacy**: Securely indexing internal HR data.

---

## ⚙️ **Workflow**
### **1. Document Preprocessing**
- **Convert all raw HR documents into PDFs.**
- **Extract text using Amazon Textract.**
- **Store processed text in an S3 bucket for further indexing.**

### **2. Semantic Chunking & Text Embedding**
- **Chunking**: Breaks text into meaningful, contextually coherent pieces for retrieval.
- **Embedding**: Uses OpenAI’s **text-embedding-3-small** model to generate vector representations.

### **3. Indexing**
- **Organizes semantic chunks in Qdrant (Vector Database)** for efficient retrieval.
- Enables **faster and more relevant search results**.

### **4. Question Classification & Retrieval** 📌 
![image](https://github.com/user-attachments/assets/94bea70f-c466-40fa-ad79-3f2b17ce2762)

- **Classifier Agent**: Categorizes input questions as:
  - **Direct Question**: Answer generated without retrieval.
  - **Simple Question**: Embeds query, retrieves relevant chunks, generates an answer.
  - **Complex Question**: Decomposes into sub-questions, retrieves & answers each sub-query, combines results.

### **5. Response Generation**
- **Retrieves relevant documents from Qdrant** based on vector similarity.
- **Generates responses** using GPT-4o.
- **Evaluates responses** using RAGAS metrics.

---

## 🏗 **AWS Infrastructure**
This project is **deployed on AWS**, ensuring high availability, security, and scalability.
![image](https://github.com/user-attachments/assets/7b6e2886-5382-4f52-ae3e-5499ecd921ec)

### **Infrastructure Components**
- **S3 Bucket**: Stores original HR documents and processed text.
- **Amazon Textract**: Extracts text from PDFs.
- **AWS Lambda (Serverless Processing)**
  - **Data Processing Lambda**: Converts raw files to PDFs and to text.
  - **Indexing Lambda**: Performs chunking & upserting into the Qdrant vector database.
  - **Backend Lambda**: Queries indexed data and generates answers.
- **API Gateway**: Routes frontend queries to the backend.
- **Qdrant (Vector Database)**: Stores embeddings for retrieval.
- **EC2 Instance (Frontend Deployment)**: Hosts the Streamlit application for user interaction.
- **VPC with Public & Private Subnets**:
  - **Public Subnet**: Hosts the EC2 frontend.
  - **Private Subnet**: Hosts the Backend Lambda for security.
- **IAM Roles & Permissions**: Restricts access to only required AWS resources.

---

## 📊 **Evaluation Metrics**
The **RAGAS Framework** is used for measuring answer quality:

| **Metric**          | **Definition**  |
|---------------------|----------------|
| **Answer Correctness** | Measures the semantic and factual similarity between the generated and ground truth answer. |
| **Faithfulness**    | Ensures responses are factually consistent with retrieved context. |
| **Context Precision** | Evaluates the relevance of retrieved context to the query. |
| **Context Recall** | Measures whether all relevant information is retrieved. |

---

## 🚀 **Productionization Approach**
To move from **PoC to Production**, we follow a **scalable and secure AWS microservices-based approach**:
- **Step 1:** **CI/CD Pipelines** - Automate deployment using AWS CDK & GitHub Actions.
- **Step 2:** **Autoscaling & Load Balancing** - Use AWS Lambda for elastic scaling and API Gateway for efficient routing.
- **Step 3:** **Security Enhancements** - Enforce IAM least privilege policies, VPC private networking, and CloudWatch monitoring.
- **Step 4:** **Fine-tuning & Optimization** - Improve indexing strategies, embedding models, and retrieval quality.

---

## 🎬 **Demo**
A **Streamlit UI** is hosted on **AWS EC2**, allowing employees to:
1. **Ask HR-related questions.**
2. **Retrieve relevant documents.**
3. **Get AI-generated responses** with citations.
4. **Get Evaluations(for Devs)**

---

## 🛠 **Setup Instructions**
### **Prerequisites**
- Python 3.9+
- Docker Installed
- AWS CLI Configured


## 📬 **Contributions**
Contributions and feedback are welcome! Feel free to open an issue or submit a pull request.

