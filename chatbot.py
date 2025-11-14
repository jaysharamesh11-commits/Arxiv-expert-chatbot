# arxiv_scientific_chatbot.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import faiss_cpu as faiss
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer
import arxiv
import os
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ArXivChatbot:
    def __init__(self):
        self.model = None
        self.index = None
        self.papers_df = None
        self.summarizer = None
        self.explanation_model = None
        self.explanation_tokenizer = None
        self.text_processor = TextProcessor()
        self.conversation_history = []
        
    def initialize_models(self):
        """Initialize all required models"""
        # Initialize sentence transformer for semantic search
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize summarization model
        if self.summarizer is None:
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    tokenizer="facebook/bart-large-cnn"
                )
            except:
                # Fallback to smaller model
                self.summarizer = pipeline("summarization", model="t5-small")
        
        # Initialize explanation model (using smaller model for demo)
        if self.explanation_model is None:
            try:
                self.explanation_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
                self.explanation_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
            except:
                st.warning("Could not load explanation model. Using simple rule-based explanations.")
    
    def load_sample_data(self):
        """Load or create sample arXiv data for demonstration"""
        sample_papers = [
            {
                'id': '2101.00001',
                'title': 'Attention Is All You Need: Transformers for Sequence Modeling',
                'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.',
                'categories': 'cs.CL cs.LG',
                'authors': ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
                'published': '2023-01-01',
                'primary_category': 'cs.CL'
            },
            {
                'id': '2101.00002',
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
                'abstract': 'We introduce a new language representation model called BERT. Our model is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.',
                'categories': 'cs.CL cs.AI',
                'authors': ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee'],
                'published': '2023-01-02',
                'primary_category': 'cs.CL'
            },
            {
                'id': '2101.00003',
                'title': 'Deep Residual Learning for Image Recognition',
                'abstract': 'Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.',
                'categories': 'cs.CV',
                'authors': ['Kaiming He', 'Xiangyu Zhang', 'Shaoqing Ren'],
                'published': '2023-01-03',
                'primary_category': 'cs.CV'
            },
            {
                'id': '2101.00004',
                'title': 'Generative Adversarial Networks',
                'abstract': 'We propose a new framework for estimating generative models via an adversarial process. We train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.',
                'categories': 'cs.LG cs.CV cs.AI',
                'authors': ['Ian Goodfellow', 'Jean Pouget-Abadie', 'Mehdi Mirza'],
                'published': '2023-01-04',
                'primary_category': 'cs.LG'
            },
            {
                'id': '2101.00005',
                'title': 'AlphaGo: Mastering the Game of Go with Deep Neural Networks and Tree Search',
                'abstract': 'The game of Go has long been viewed as the most challenging of classic games for artificial intelligence. We introduce a new approach to computer Go that uses value networks to evaluate board positions and policy networks to select moves.',
                'categories': 'cs.AI',
                'authors': ['David Silver', 'Aja Huang', 'Chris Maddison'],
                'published': '2023-01-05',
                'primary_category': 'cs.AI'
            }
        ]
        
        self.papers_df = pd.DataFrame(sample_papers)
        self.papers_df['abstract_clean'] = self.papers_df['abstract'].str.replace('\n', ' ').str.strip()
        self.papers_df['title_clean'] = self.papers_df['title'].str.strip()
        
        return self.papers_df
    
    def build_search_index(self):
        """Build FAISS index for semantic search"""
        if self.papers_df is None:
            self.load_sample_data()
        
        abstracts = self.papers_df['abstract_clean'].tolist()
        embeddings = self.model.encode(abstracts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def search_papers(self, query, k=5):
        """Search for relevant papers"""
        if self.index is None:
            self.build_search_index()
        
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.papers_df):
                paper = self.papers_df.iloc[idx]
                results.append({
                    'index': idx,
                    'score': float(score),
                    'title': paper['title_clean'],
                    'abstract': paper['abstract_clean'],
                    'authors': paper['authors'],
                    'categories': paper['categories'],
                    'published': paper['published'],
                    'primary_category': paper['primary_category']
                })
        
        return results
    
    def summarize_paper(self, abstract, max_length=150):
        """Generate summary of paper abstract"""
        try:
            summary = self.summarizer(
                abstract,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            return summary
        except:
            # Fallback: return first few sentences
            sentences = sent_tokenize(abstract)
            return ' '.join(sentences[:2])
    
    def generate_explanation(self, concept, context=""):
        """Generate explanation for a concept"""
        prompt = f"Explain the concept of {concept} in machine learning and artificial intelligence:"
        
        if self.explanation_model:
            try:
                inputs = self.explanation_tokenizer.encode(prompt + context, return_tensors='pt')
                outputs = self.explanation_model.generate(
                    inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.explanation_tokenizer.eos_token_id
                )
                explanation = self.explanation_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return explanation.replace(prompt, "").strip()
            except:
                pass
        
        # Rule-based fallback explanations
        explanations = {
            "transformer": "Transformers are deep learning models that use self-attention mechanisms to process sequential data. Unlike RNNs, they process all elements in parallel, making them highly efficient. Key components include multi-head attention and positional encoding.",
            "bert": "BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model pre-trained on large text corpora using masked language modeling and next sentence prediction objectives.",
            "residual network": "Residual Networks (ResNets) use skip connections to address the vanishing gradient problem in deep networks, enabling training of very deep neural networks.",
            "gan": "Generative Adversarial Networks consist of two neural networks - generator and discriminator - that compete against each other, enabling realistic data generation.",
            "attention": "Attention mechanisms allow models to focus on relevant parts of input data, weighing the importance of different elements when making predictions."
        }
        
        concept_lower = concept.lower()
        for key, explanation in explanations.items():
            if key in concept_lower:
                return explanation
        
        return f"I can explain {concept} based on available research. This appears to be a machine learning concept. Would you like me to search for specific papers about it?"
    
    def handle_query(self, user_input, conversation_context):
        """Main method to handle user queries"""
        self.initialize_models()
        
        # Add to conversation history
        self.conversation_history.append({"user": user_input, "timestamp": datetime.now()})
        
        response = {
            "answer": "",
            "papers": [],
            "concepts": [],
            "visualization_data": None,
            "follow_up_questions": []
        }
        
        # Detect query type
        query_type = self.classify_query(user_input)
        
        if query_type == "paper_search":
            papers = self.search_papers(user_input)
            response["papers"] = papers
            
            if papers:
                paper_titles = [p['title'] for p in papers[:3]]
                response["answer"] = f"I found {len(papers)} relevant papers. Here are the top results:\n\n" + "\n".join([f"• {title}" for title in paper_titles])
                
                # Generate follow-up questions
                response["follow_up_questions"] = [
                    f"Can you summarize '{papers[0]['title']}'?",
                    "Show me visualization of paper categories",
                    "Explain the main concept of the first paper"
                ]
            else:
                response["answer"] = "No relevant papers found. Try rephrasing your query."
        
        elif query_type == "concept_explanation":
            concepts = self.text_processor.extract_key_concepts(user_input)
            if concepts:
                main_concept = concepts[0]
                explanation = self.generate_explanation(main_concept, user_input)
                response["answer"] = f"**Explanation of {main_concept}:**\n\n{explanation}"
                response["concepts"] = concepts
                
                # Find related papers
                related_papers = self.search_papers(main_concept, k=3)
                response["papers"] = related_papers
                
                response["follow_up_questions"] = [
                    f"Show me papers about {main_concept}",
                    f"How does {main_concept} relate to {concepts[1] if len(concepts) > 1 else 'other ML concepts'}?",
                    "Give me a simpler explanation"
                ]
        
        elif query_type == "summarization":
            # Extract paper title from query
            papers = self.search_papers(user_input, k=1)
            if papers:
                summary = self.summarize_paper(papers[0]['abstract'])
                response["answer"] = f"**Summary of '{papers[0]['title']}':**\n\n{summary}"
                response["papers"] = [papers[0]]
        
        else:
            response["answer"] = "I can help you with:\n• Searching research papers\n• Explaining ML concepts\n• Summarizing papers\n\nTry asking about specific topics like 'transformers', 'neural networks', or 'search for papers about GANs'."
        
        return response

    def classify_query(self, query):
        """Classify the type of user query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['search', 'find', 'paper', 'research', 'article']):
            return "paper_search"
        elif any(word in query_lower for word in ['explain', 'what is', 'how does', 'concept', 'meaning']):
            return "concept_explanation"
        elif any(word in query_lower for word in ['summarize', 'summary', 'brief', 'overview']):
            return "summarization"
        else:
            return "general"

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def extract_key_concepts(self, text, max_concepts=5):
        """Extract key concepts from text"""
        words = re.findall(r'\b[a-zA-Z][a-zA-Z-]+\b', text.lower())
        concepts = [word for word in words if word not in self.stop_words and len(word) > 3]
        
        from collections import Counter
        return [concept for concept, count in Counter(concepts).most_common(max_concepts)]

class VisualizationEngine:
    @staticmethod
    def create_category_chart(papers):
        """Create visualization of paper categories"""
        if not papers:
            return None
        
        categories = []
        for paper in papers:
            if 'categories' in paper:
                cats = paper['categories'].split()
                categories.extend(cats)
        
        if not categories:
            return None
            
        from collections import Counter
        category_counts = Counter(categories)
        
        fig = px.bar(
            x=list(category_counts.keys()),
            y=list(category_counts.values()),
            title="Paper Categories Distribution",
            labels={'x': 'Category', 'y': 'Count'}
        )
        
        return fig
    
    @staticmethod
    def create_similarity_plot(papers, query):
        """Create similarity score visualization"""
        if not papers:
            return None
        
        titles = [paper['title'][:50] + '...' for paper in papers]
        scores = [paper['score'] for paper in papers]
        
        fig = px.bar(
            x=scores,
            y=titles,
            orientation='h',
            title=f"Similarity Scores for: '{query}'",
            labels={'x': 'Similarity Score', 'y': 'Papers'}
        )
        
        return fig

def main():
    st.set_page_config(
        page_title="arXiv Scientific Chatbot",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .paper-card {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .concept-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.2rem;
        background-color: #e3f2fd;
        border-radius: 1rem;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧠 arXiv Scientific Chatbot</h1>', unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ArXivChatbot()
        st.session_state.chatbot.initialize_models()
        st.session_state.chatbot.load_sample_data()
        st.session_state.chatbot.build_search_index()
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        st.subheader("Search Options")
        search_category = st.selectbox(
            "Filter by Category",
            ["All", "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE"]
        )
        
        num_results = st.slider("Number of Results", 1, 10, 5)
        
        st.subheader("Visualization")
        show_viz = st.checkbox("Show Visualizations", value=True)
        
        st.markdown("---")
        st.subheader("💡 Example Queries")
        example_queries = [
            "Search for papers about transformers",
            "Explain attention mechanism",
            "What are generative adversarial networks?",
            "Summarize recent computer vision papers",
            "Find research on neural networks"
        ]
        
        for query in example_queries:
            if st.button(f"\"{query}\"", key=query):
                st.session_state.user_input = query
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "papers" in message and message["papers"]:
                    with st.expander("📄 Related Papers"):
                        for i, paper in enumerate(message["papers"][:3]):
                            st.markdown(f"""
                            <div class="paper-card">
                                <strong>{paper['title']}</strong><br>
                                <em>Authors: {', '.join(paper['authors'][:3])}</em><br>
                                <small>Categories: {paper['categories']}</small><br>
                                <small>Similarity: {paper['score']:.3f}</small>
                            </div>
                            """, unsafe_allow_html=True)
        
        # User input
        user_input = st.chat_input("Ask about research papers or ML concepts...")
        
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get chatbot response
            with st.spinner("Searching papers and generating response..."):
                response = st.session_state.chatbot.handle_query(user_input, st.session_state.messages)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
                
                # Display related papers
                if response["papers"]:
                    with st.expander("📚 Relevant Research Papers"):
                        for i, paper in enumerate(response["papers"]):
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.markdown(f"**{paper['title']}**")
                                st.caption(f"Authors: {', '.join(paper['authors'][:3])}")
                                st.caption(f"Categories: {paper['categories']}")
                            with col_b:
                                st.metric("Similarity", f"{paper['score']:.3f}")
                
                # Display key concepts
                if response["concepts"]:
                    st.markdown("**🔑 Key Concepts:**")
                    concepts_html = "".join([f'<span class="concept-badge">{concept}</span>' for concept in response["concepts"]])
                    st.markdown(concepts_html, unsafe_allow_html=True)
                
                # Display follow-up questions
                if response["follow_up_questions"]:
                    st.markdown("**🤔 Follow-up Questions:**")
                    for question in response["follow_up_questions"]:
                        if st.button(question, key=question):
                            st.session_state.user_input = question
            
            # Add assistant response to messages with papers data
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["answer"],
                "papers": response["papers"]
            })
    
    with col2:
        st.subheader("📊 Visualizations")
        
        if st.session_state.messages and show_viz:
            last_assistant_msg = None
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant" and "papers" in msg and msg["papers"]:
                    last_assistant_msg = msg
                    break
            
            if last_assistant_msg:
                viz_engine = VisualizationEngine()
                
                # Category distribution
                category_fig = viz_engine.create_category_chart(last_assistant_msg["papers"])
                if category_fig:
                    st.plotly_chart(category_fig, use_container_width=True)
                
                # Similarity scores
                if len(st.session_state.messages) >= 2:
                    user_query = st.session_state.messages[-2]["content"]
                    similarity_fig = viz_engine.create_similarity_plot(
                        last_assistant_msg["papers"], 
                        user_query
                    )
                    if similarity_fig:
                        st.plotly_chart(similarity_fig, use_container_width=True)
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("🔄 Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("📚 Show Sample Papers"):
            sample_query = "Show me sample machine learning papers"
            st.session_state.user_input = sample_query

if __name__ == "__main__":
    main()