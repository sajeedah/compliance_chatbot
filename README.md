# compliance\_chatbot

AI-powered Compliance Assistant (FATF + VARA RAG)



An AI-powered Compliance Assistant that answers regulatory questions such as:



\- \*“What does FATF Recommendation 16 say about wire transfers?”\*

\- \*“How must Client Money be safeguarded under VARA?”\*



This chatbot uses a Retrieval-Augmented Generation (RAG) pipeline to search official rulebooks and provide:

✅ Bullet-point answers  

✅ Short summary  

✅ Verbatim quote (≤ 30 words)  

✅ Proper citations (doc + section)



If the rule is not found → it replies \*\*“Insufficient context.”\*\*



---



\## 🚀 Key Features



\- FATF + VARA rule-aware responses

\- Citation-based and explainable output

\- Guardrails against hallucination

\- Fully local document control (PDF/Markdown)

\- Mini Web App built with \*\*Streamlit\*\*



---



\## 🗂️ Project Structure



compliance\_chatbot/

├─ app.py # Streamlit Web App

├─ requirements.txt

├─ README.md

├─ src/ # RAG + retrieval modules

├─ docs/ # FATF + VARA source files

└─ tests/ # Evaluation set



yaml

Copy code



---



\## ⚙️ Run Locally



```bash

pip install -r requirements.txt

streamlit run app.py

Create a .env file in the project root and add:



ini

Copy code

OPENAI\_API\_KEY=sk-xxxx

## 🌐 Live App

🔗 **ComplianceBot (FATF + VARA)**: https://sajeedah-compliance-chatbot.streamlit.app/



👨‍💼 Author

Mohamed Sajeed

Compliance Officer | Data \& Analytics | AML Enthusiast

🔗 LinkedIn



⭐ If you like this project, please star the repository!

