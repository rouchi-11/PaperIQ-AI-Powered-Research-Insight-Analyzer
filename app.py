# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import PyPDF2
import io
import random
from datetime import datetime

# Fix SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download all required NLTK data with error handling
nltk_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
for package in nltk_packages:
    try:
        if 'punkt' in package:
            nltk.data.find(f'tokenizers/{package}')
        elif package in ['stopwords', 'wordnet']:
            nltk.data.find(f'corpora/{package}')
        else:
            nltk.data.find(f'taggers/{package}')
    except LookupError:
        print(f"Downloading NLTK {package}...")
        nltk.download(package, quiet=True)

# Set page config
st.set_page_config(
    page_title="PaperIQ AI Powered Research Insight Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful, modern CSS with appealing colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        padding: 1rem;
    }
    
    .section-header {
        font-size: 1.6rem;
        color: #2d3748;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-bottom: 3px solid;
        border-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%) 1;
        padding-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #4a5568;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    .entity-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.3rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.85rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: none;
    }
    
    .entity-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .method { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }
    
    .dataset { 
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
    }
    
    .technology { 
        background: linear-gradient(135deg, #45b7d1 0%, #96c93d 100%);
        color: white;
    }
    
    .metric { 
        background: linear-gradient(135deg, #b06ab3 0%, #4568dc 100%);
        color: white;
    }
    
    .highlight {
        padding: 0.3rem 0.5rem;
        border-radius: 8px;
        font-weight: 600;
        margin: 0 0.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .highlight-method { 
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        color: #2d3436;
        border: 1px solid #fd79a8;
    }
    
    .highlight-tech { 
        background: linear-gradient(135deg, #81ecec 0%, #74b9ff 100%);
        color: #2d3436;
        border: 1px solid #0984e3;
    }
    
    .highlight-dataset { 
        background: linear-gradient(135deg, #55efc4 0%, #00b894 100%);
        color: #2d3436;
        border: 1px solid #00b894;
    }
    
    .highlight-metric { 
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        color: white;
        border: 1px solid #6c5ce7;
    }
    
    .chat-message {
        padding: 1.2rem;
        border-radius: 15px;
        margin-bottom: 1.2rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: none;
    }
    
    .chat-message:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin-left: 15%;
        color: white;
        border-left: 5px solid #4a90e2;
    }
    
    .chat-message.assistant {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        margin-right: 15%;
        color: white;
        border-left: 5px solid #e84393;
    }
    
    .preview-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: none;
        margin: 1rem 0;
        line-height: 1.7;
        font-size: 0.95rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stat-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: none;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stat-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #718096;
        font-weight: 500;
    }
    
    .tab-content {
        padding: 1.5rem 0;
    }
    
    .analysis-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: none;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .analysis-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .analysis-card h4 {
        margin-top: 0;
        color: #2d3748;
        border-bottom: 2px solid;
        border-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%) 1;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stTextInput input, .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.7rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSelectbox select {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .quick-question {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .quick-question:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #fed6e3 0%, #a8edea 100%);
    }
    
    .feature-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_papers' not in st.session_state:
    st.session_state.processed_papers = []
if 'current_paper' not in st.session_state:
    st.session_state.current_paper = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'qa_processor' not in st.session_state:
    st.session_state.qa_processor = None
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = True

# Enhanced Document Understanding System
class IntelligentDocumentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Comprehensive entity patterns for research papers
        self.entity_patterns = {
            'RESEARCH_METHOD': [
                # AI/ML Methods
                r'\b(?:deep learning|machine learning|neural networks?|artificial intelligence|ai|ml|dl)\b',
                r'\b(?:transformer|bert|gpt|resnet|vit|vision transformer|attention mechanism)\b',
                r'\b(?:cnn|rnn|lstm|gru|convolutional|recurrent|feedforward)\b',
                r'\b(?:supervised|unsupervised|semi-supervised|reinforcement learning)\b',
                r'\b(?:classification|regression|clustering|dimensionality reduction)\b',
                
                # Research Methods
                r'\b(?:experimental|empirical|theoretical|computational|statistical)\s+(?:analysis|study|approach|method)\b',
                r'\b(?:case study|survey|literature review|meta-analysis|systematic review)\b',
                r'\b(?:qualitative|quantitative|mixed methods?)\s+(?:research|analysis|approach)\b',
                
                # Algorithms
                r'\b(?:algorithm|svm|random forest|decision tree|k-means|k-nn|naive bayes)\b',
                r'\b(?:gradient descent|backpropagation|optimization|feature selection)\b',
                
                # Knowledge Systems
                r'\b(?:knowledge.based|expert system|rule.based|ontology|semantic)\b',
                r'\b(?:reasoning|inference|knowledge representation|knowledge graph)\b'
            ],
            
            'TECHNOLOGY': [
                r'\b(?:pytorch|tensorflow|keras|scikit-learn|pandas|numpy)\b',
                r'\b(?:python|r|java|matlab|sql|mongodb|elasticsearch)\b',
                r'\b(?:gpu|cuda|cloud computing|aws|azure|google cloud)\b',
                r'\b(?:api|rest|json|xml|csv|database|big data)\b',
                r'\b(?:framework|library|platform|tool|software|system)\b'
            ],
            
            'DATASET': [
                r'\b(?:imagenet|mnist|cifar|coco|pascal voc|wmt|glue|squad)\b',
                r'\b(?:dataset|corpus|benchmark|training data|test data)\b',
                r'\b(?:wikidata|dbpedia|freebase|conceptnet|wordnet)\b',
                r'\b(?:pubmed|arxiv|google scholar|academic papers)\b',
                r'\b[A-Z][A-Za-z0-9\-_]*\s+(?:dataset|corpus|benchmark|collection)\b'
            ],
            
            'METRIC': [
                r'\b(?:accuracy|precision|recall|f1.score|f1|bleu|rouge)\b',
                r'\b(?:auc|roc|mse|mae|rmse|r2|correlation)\b',
                r'\b(?:perplexity|cer|wer|meteor|chrf|bertscore)\b',
                r'\b\d+\.?\d*\s*%?\s*(?:accuracy|precision|recall|f1|score)\b',
                r'\b(?:top.1|top.5)\s+accuracy\b'
            ]
        }
        
        # Domain-specific knowledge base
        self.knowledge_base = {}
        
    def comprehensive_analysis(self, text):
        """Perform comprehensive document analysis"""
        analysis = {
            'entities': self.extract_entities_advanced(text),
            'topics': self.extract_topics(text),
            'key_concepts': self.extract_key_concepts(text),
            'research_questions': self.identify_research_questions(text),
            'methodology': self.extract_methodology(text),
            'findings': self.extract_findings(text),
            'document_summary': self.generate_summary(text),
            'confidence_scores': self.calculate_confidence_scores(text)
        }
        return analysis
    
    def extract_entities_advanced(self, text):
        """Advanced entity extraction with context awareness"""
        entities = defaultdict(set)
        
        # POS tagging for better accuracy
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Apply regex patterns
            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        entity = match.group().strip()
                        if len(entity) > 2 and entity.lower() not in self.stop_words:
                            entities[entity_type].add(entity)
            
            # Extract noun phrases as potential entities
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            noun_phrases = self._extract_noun_phrases(pos_tags)
            for phrase in noun_phrases:
                phrase_lower = phrase.lower()
                if any(keyword in phrase_lower for keyword in ['system', 'method', 'approach', 'model', 'algorithm']):
                    entities['RESEARCH_METHOD'].add(phrase)
                elif any(keyword in phrase_lower for keyword in ['dataset', 'data', 'corpus', 'benchmark']):
                    entities['DATASET'].add(phrase)
        
        # Clean and validate entities
        cleaned_entities = {}
        for entity_type, entity_set in entities.items():
            cleaned_entities[entity_type] = self._clean_entities(list(entity_set))
        
        return cleaned_entities
    
    def _extract_noun_phrases(self, pos_tags):
        """Extract noun phrases from POS tagged text"""
        noun_phrases = []
        current_phrase = []
        
        for word, tag in pos_tags:
            if tag.startswith('NN') or tag.startswith('JJ') or word.lower() in ['based', 'using', 'with']:
                current_phrase.append(word)
            else:
                if len(current_phrase) >= 2:
                    phrase = ' '.join(current_phrase)
                    if len(phrase) > 5:
                        noun_phrases.append(phrase)
                current_phrase = []
        
        if len(current_phrase) >= 2:
            noun_phrases.append(' '.join(current_phrase))
        
        return noun_phrases
    
    def _clean_entities(self, entity_list):
        """Clean and validate extracted entities"""
        cleaned = []
        for entity in entity_list:
            entity = re.sub(r'[^\w\s\-\.]', '', entity).strip()
            entity = ' '.join(entity.split())
            
            if (len(entity) >= 3 and len(entity) <= 100 and 
                not entity.lower() in self.stop_words and
                not entity.isdigit()):
                cleaned.append(entity)
        
        return list(set(cleaned))[:10]  # Limit to top 10 per category
    
    def extract_topics(self, text):
        """Extract main topics using LDA"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 5:
                return []
            
            vectorizer = CountVectorizer(max_features=100, stop_words='english', 
                                       ngram_range=(1, 2), min_df=1)
            doc_matrix = vectorizer.fit_transform(sentences)
            
            lda = LatentDirichletAllocation(n_components=3, random_state=42, max_iter=10)
            lda.fit(doc_matrix)
            
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-5:][::-1]]
                topics.append(' '.join(top_words))
            
            return topics
        except:
            return []
    
    def extract_key_concepts(self, text):
        """Extract key concepts and their importance"""
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english', 
                                   ngram_range=(1, 3), min_df=1)
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            concept_scores = list(zip(feature_names, scores))
            concept_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [concept for concept, score in concept_scores[:15] if score > 0.1]
        except:
            return []
    
    def identify_research_questions(self, text):
        """Identify research questions and objectives"""
        question_patterns = [
            r'(?:research question|question|objective|aim|goal|purpose).*?[.?]',
            r'(?:we aim to|this paper|this study|our goal).*?[.]',
            r'(?:investigate|examine|explore|analyze|study).*?[.]'
        ]
        
        questions = []
        for pattern in question_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                question = match.group().strip()
                if len(question) > 20 and len(question) < 200:
                    questions.append(question)
        
        return questions[:3]  # Return top 3
    
    def extract_methodology(self, text):
        """Extract methodology information"""
        method_indicators = [
            r'(?:method|approach|technique|procedure|algorithm).*?[.]',
            r'(?:we use|we employ|we apply|we implement).*?[.]',
            r'(?:our approach|our method|our system).*?[.]'
        ]
        
        methodology = []
        for pattern in method_indicators:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                method = match.group().strip()
                if len(method) > 15 and len(method) < 150:
                    methodology.append(method)
        
        return methodology[:5]
    
    def extract_findings(self, text):
        """Extract key findings and results"""
        result_patterns = [
            r'(?:result|finding|conclusion|achieve|obtain|show).*?[.]',
            r'(?:accuracy|precision|recall|f1|score).*?[.]',
            r'(?:our model|our approach|our system).*?(?:achieve|obtain|show).*?[.]'
        ]
        
        findings = []
        for pattern in result_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                finding = match.group().strip()
                if len(finding) > 15 and len(finding) < 150:
                    findings.append(finding)
        
        return findings[:5]
    
    def generate_summary(self, text):
        """Generate an intelligent summary of the document"""
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return text
        
        # Simple extractive summarization using TF-IDF
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores
            sentence_scores = np.mean(tfidf_matrix.toarray(), axis=1)
            top_indices = np.argsort(sentence_scores)[-3:][::-1]
            
            summary_sentences = [sentences[i] for i in sorted(top_indices)]
            return ' '.join(summary_sentences)
        except:
            return ' '.join(sentences[:3])
    
    def calculate_confidence_scores(self, text):
        """Calculate confidence scores for different aspects"""
        word_count = len(text.split())
        
        scores = {
            'completeness': min(1.0, word_count / 1000),
            'technical_depth': self._calculate_technical_depth(text),
            'entity_coverage': self._calculate_entity_coverage(text),
            'structure_quality': self._calculate_structure_quality(text)
        }
        
        return scores
    
    def _calculate_technical_depth(self, text):
        """Calculate technical depth based on domain-specific terms"""
        # Use entity patterns to determine technical depth
        total_patterns = sum(len(patterns) for patterns in self.entity_patterns.values())
        found_patterns = 0
        
        for patterns in self.entity_patterns.values():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found_patterns += 1
        
        return found_patterns / total_patterns if total_patterns > 0 else 0
    
    def _calculate_entity_coverage(self, text):
        """Calculate entity coverage score"""
        total_patterns = sum(len(patterns) for patterns in self.entity_patterns.values())
        found_patterns = 0
        
        for patterns in self.entity_patterns.values():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found_patterns += 1
        
        return found_patterns / total_patterns if total_patterns > 0 else 0
    
    def _calculate_structure_quality(self, text):
        """Calculate document structure quality"""
        sentences = sent_tokenize(text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        
        # Ideal sentence length is between 15-25 words
        if 15 <= avg_sentence_length <= 25:
            return 1.0
        else:
            return max(0.3, 1.0 - abs(avg_sentence_length - 20) / 20)

# Intelligent Question Answering System
class IntelligentQASystem:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 3))
        self.sentences = []
        self.sentence_embeddings = None
        self.document_analysis = None
        
        # Enhanced question understanding patterns
        self.question_types = {
            'what_is': [r'what is', r'what are', r'define', r'definition of'],
            'how': [r'how', r'how to', r'how does', r'how do'],
            'why': [r'why', r'reason', r'because', r'purpose'],
            'when': [r'when', r'time', r'date', r'period'],
            'where': [r'where', r'location', r'place'],
            'which': [r'which', r'what type', r'what kind'],
            'comparison': [r'compare', r'difference', r'better', r'versus', r'vs'],
            'methodology': [r'method', r'approach', r'technique', r'algorithm', r'process'],
            'results': [r'result', r'outcome', r'finding', r'performance', r'accuracy'],
            'technology': [r'technology', r'tool', r'framework', r'system', r'platform'],
            'dataset': [r'dataset', r'data', r'corpus', r'benchmark', r'training'],
        }
        
    def initialize(self, text, document_analysis):
        """Initialize the QA system with document content and analysis"""
        self.document_analysis = document_analysis
        
        # Prepare sentences for vector search
        sentences = sent_tokenize(text)
        cleaned_sentences = []
        
        for sentence in sentences:
            cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?%-]', '', sentence)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned and len(cleaned.split()) > 3:
                cleaned_sentences.append(cleaned)
        
        self.sentences = cleaned_sentences
        
        if self.sentences:
            try:
                self.sentence_embeddings = self.vectorizer.fit_transform(self.sentences)
                return True
            except:
                return False
        return False
    
    def answer_question(self, question):
        """Generate comprehensive answer to user question"""
        if not self.sentences or not self.document_analysis:
            return "Please analyze a document first."
        
        question_lower = question.lower().strip()
        if not question_lower:
            return "Please enter a valid question."
        
        # Step 1: Classify question type
        question_type = self._classify_question(question_lower)
        
        # Step 2: Generate answer based on question type
        if question_type == 'what_is':
            return self._answer_definition_question(question_lower)
        elif question_type == 'methodology':
            return self._answer_methodology_question(question_lower)
        elif question_type == 'results':
            return self._answer_results_question(question_lower)
        elif question_type == 'technology':
            return self._answer_technology_question(question_lower)
        elif question_type == 'dataset':
            return self._answer_dataset_question(question_lower)
        elif question_type == 'comparison':
            return self._answer_comparison_question(question_lower)
        else:
            return self._answer_general_question(question_lower)
    
    def _classify_question(self, question):
        """Classify the type of question being asked"""
        for q_type, patterns in self.question_types.items():
            if any(pattern in question for pattern in patterns):
                return q_type
        return 'general'
    
    def _get_document_context(self, question):
        """Get relevant context from the current document"""
        try:
            question_vec = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, self.sentence_embeddings)[0]
            
            if np.max(similarities) > 0.2:
                best_idx = np.argmax(similarities)
                return self.sentences[best_idx]
        except:
            pass
        return None
    
    def _answer_definition_question(self, question):
        """Answer definition questions"""
        # Extract the concept being asked about
        concept_patterns = [
            r'what is (.*?)[\?\.]',
            r'what are (.*?)[\?\.]',
            r'define (.*?)[\?\.]',
            r'definition of (.*?)[\?\.]'
        ]
        
        concept = None
        for pattern in concept_patterns:
            match = re.search(pattern, question)
            if match:
                concept = match.group(1).strip()
                break
        
        if concept:
            # Look for concept in entities
            entities = self.document_analysis.get('entities', {})
            
            # Check if concept appears in any entity category
            for category, entity_list in entities.items():
                for entity in entity_list:
                    if concept.lower() in entity.lower() or entity.lower() in concept.lower():
                        category_desc = {
                            'RESEARCH_METHOD': 'a research methodology or approach',
                            'TECHNOLOGY': 'a technology or framework',
                            'DATASET': 'a dataset or data source',
                            'METRIC': 'a performance metric or measure'
                        }
                        
                        context = self._get_document_context(concept)
                        base_answer = f"{entity} is {category_desc.get(category, 'an important concept')} mentioned in this research."
                        
                        if context:
                            return f"{base_answer} {context}"
                        return base_answer
        
        # Fallback to semantic search
        return self._semantic_search_answer(question)
    
    def _answer_methodology_question(self, question):
        """Answer methodology-related questions"""
        methods = self.document_analysis.get('entities', {}).get('RESEARCH_METHOD', [])
        methodology = self.document_analysis.get('methodology', [])
        
        response = "The research methodology includes:\n"
        
        if methods:
            response += f"Methods: {', '.join(methods[:5])}\n"
        
        if methodology:
            response += f"Approach: {methodology[0]}"
        else:
            response += self._semantic_search_answer(question)
        
        return response
    
    def _answer_results_question(self, question):
        """Answer results and findings questions"""
        metrics = self.document_analysis.get('entities', {}).get('METRIC', [])
        findings = self.document_analysis.get('findings', [])
        
        response = "Key results and findings:\n"
        
        if metrics:
            response += f"Performance metrics: {', '.join(metrics[:5])}\n"
        
        if findings:
            response += f"Main findings: {findings[0]}"
        else:
            response += self._semantic_search_answer(question)
        
        return response
    
    def _answer_technology_question(self, question):
        """Answer technology-related questions"""
        technologies = self.document_analysis.get('entities', {}).get('TECHNOLOGY', [])
        
        if technologies:
            response = f"Technologies and frameworks used: {', '.join(technologies[:5])}"
            context = self._get_document_context(question)
            if context:
                response += f"\n\nAdditional context: {context}"
            return response
        
        return self._semantic_search_answer(question)
    
    def _answer_dataset_question(self, question):
        """Answer dataset-related questions"""
        datasets = self.document_analysis.get('entities', {}).get('DATASET', [])
        
        if datasets:
            response = f"Datasets used: {', '.join(datasets[:5])}"
            context = self._get_document_context(question)
            if context:
                response += f"\n\nContext: {context}"
            return response
        
        return self._semantic_search_answer(question)
    
    def _answer_comparison_question(self, question):
        """Answer comparison questions"""
        context = self._get_document_context(question)
        if context:
            return f"Regarding comparisons: {context}"
        
        return "The document discusses comparisons, but I need more specific information about what you'd like to compare."
    
    def _answer_general_question(self, question):
        """Answer general questions using semantic search"""
        return self._semantic_search_answer(question)
    
    def _semantic_search_answer(self, question):
        """Use semantic search to find relevant answers"""
        try:
            question_vec = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, self.sentence_embeddings)[0]
            
            # Get top 3 most relevant sentences
            top_indices = np.argsort(similarities)[-3:][::-1]
            relevant_sentences = []
            
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    relevant_sentences.append(self.sentences[idx])
            
            if relevant_sentences:
                return ' '.join(relevant_sentences)
            else:
                # Fallback to document summary
                summary = self.document_analysis.get('document_summary', '')
                if summary:
                    return f"Based on the document: {summary}"
                
                return "I couldn't find a specific answer. The document covers various research topics and methodologies."
        
        except Exception as e:
            return f"I encountered an issue processing your question. Please try rephrasing it."

# PDF Processing Function
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Initialize systems
document_analyzer = IntelligentDocumentAnalyzer()

# Main Application
def main():
    st.markdown('<h1 class="main-header">🔬 PaperIQ AI Powered Research Insight Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #718096; font-size: 1.2rem; margin-bottom: 2rem;">Intelligent Document Analysis & Research Assistant</p>', unsafe_allow_html=True)
    
    # Initialize QA processor in session state if not exists
    if st.session_state.qa_processor is None:
        st.session_state.qa_processor = IntelligentQASystem()
    
    # File upload and text input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="section-header">📄 Document Input</h3>', unsafe_allow_html=True)
        input_method = st.radio("Choose input method:", 
                               ("Upload PDF", "Paste Text"),
                               horizontal=True)
        
        text = ""
        paper_title = ""
        
        if input_method == "Upload PDF":
            uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
            if uploaded_file is not None:
                with st.spinner("Extracting and analyzing text from PDF..."):
                    text = extract_text_from_pdf(uploaded_file)
                    paper_title = uploaded_file.name
                    st.success("✅ PDF processed successfully!")
        else:
            text = st.text_area("Paste your research paper text:", 
                               height=200,
                               placeholder="Enter research paper text here...",
                               value="We propose a novel Knowledge-Based System that integrates deep learning with symbolic reasoning for enhanced artificial intelligence applications. Our approach combines neural networks with ontological knowledge representation to achieve superior performance in natural language processing tasks. The system was evaluated on multiple datasets including GLUE benchmark and achieved 94.2% accuracy, outperforming traditional machine learning methods by 12.3%. We implemented our solution using PyTorch and TensorFlow frameworks, leveraging GPU acceleration for efficient training. The knowledge graph contains over 1 million entities and relationships, enabling comprehensive reasoning capabilities. Our findings demonstrate that hybrid AI systems combining symbolic and connectionist approaches can significantly improve performance on complex cognitive tasks.")
            paper_title = "User Input Text"
        
        # Analysis button
        if text and st.button("🚀 Analyze Document", type="primary", use_container_width=True):
            with st.spinner("Performing comprehensive document analysis..."):
                # Comprehensive analysis
                analysis = document_analyzer.comprehensive_analysis(text)
                
                # Initialize QA system
                qa_success = st.session_state.qa_processor.initialize(text, analysis)
                
                paper_data = {
                    'title': paper_title,
                    'text': text,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat(),
                    'qa_ready': qa_success
                }
                
                st.session_state.current_paper = paper_data
                st.session_state.processed_papers.append(paper_data)
                st.session_state.show_analysis = True
                st.session_state.chat_history = []  # Reset chat for new document
                
                success_msg = "✅ Comprehensive analysis complete! "
                success_msg += "🤖 Intelligent QA system ready." if qa_success else "📊 Basic QA available."
                st.success(success_msg)
                st.rerun()
    
    # Document Analysis Display
    if st.session_state.current_paper and st.session_state.show_analysis:
        st.markdown("---")
        st.markdown('<h2 class="section-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
        
        analysis = st.session_state.current_paper['analysis']
        
        # Analysis Results Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Key Insights", "🏷️ Entities", "📝 Summary", "⚙️ Methodology"])
        
        with tab1:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.markdown('<div class="analysis-card"><h4>🎯 Main Topics</h4>', unsafe_allow_html=True)
                topics = analysis.get('topics', [])
                if topics:
                    for i, topic in enumerate(topics):
                        st.markdown(f"**Topic {i+1}:** {topic}")
                else:
                    st.info("Topics will be extracted from longer documents.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="analysis-card"><h4>💡 Key Concepts</h4>', unsafe_allow_html=True)
                concepts = analysis.get('key_concepts', [])[:10]
                if concepts:
                    for concept in concepts:
                        st.markdown(f"• {concept}")
                else:
                    st.info("Key concepts not identified.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_insight2:
                st.markdown('<div class="analysis-card"><h4>❓ Research Questions</h4>', unsafe_allow_html=True)
                questions = analysis.get('research_questions', [])
                if questions:
                    for question in questions:
                        st.markdown(f"• {question}")
                else:
                    st.info("Research questions not clearly identified.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="analysis-card"><h4>📈 Key Findings</h4>', unsafe_allow_html=True)
                findings = analysis.get('findings', [])
                if findings:
                    for finding in findings:
                        st.markdown(f"• {finding}")
                else:
                    st.info("Key findings not extracted.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            entities = analysis.get('entities', {})
            
            # Entity visualization
            entity_counts = {k.replace('_', ' ').title(): len(v) for k, v in entities.items() if v}
            if entity_counts:
                fig = px.bar(
                    x=list(entity_counts.keys()),
                    y=list(entity_counts.values()),
                    title="📊 Entities by Category",
                    color=list(entity_counts.values()),
                    color_continuous_scale="viridis"
                )
                fig.update_layout(
                    height=350, 
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#2d3748")
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Entity details with beautiful badges
            for entity_type, entity_list in entities.items():
                if entity_list:
                    st.markdown(f"<h4>{entity_type.replace('_', ' ').title()}</h4>", unsafe_allow_html=True)
                    for entity in entity_list[:8]:  # Show top 8
                        class_name = {
                            'RESEARCH_METHOD': 'method',
                            'TECHNOLOGY': 'technology', 
                            'DATASET': 'dataset',
                            'METRIC': 'metric'
                        }.get(entity_type, 'method')
                        st.markdown(f'<span class="entity-badge {class_name}">{entity}</span>', 
                                  unsafe_allow_html=True)
                    st.markdown("")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            
            st.markdown('<div class="analysis-card"><h4>📋 Document Summary</h4>', unsafe_allow_html=True)
            summary = analysis.get('document_summary', '')
            if summary:
                st.markdown(f'<div class="preview-box">{summary}</div>', unsafe_allow_html=True)
            else:
                st.info("Summary not available for short texts.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Highlighted text
            st.markdown('<div class="analysis-card"><h4>🎨 Entity-Highlighted Text</h4>', unsafe_allow_html=True)
            entities = analysis.get('entities', {})
            highlighted_text = highlight_entities_adaptive(
                st.session_state.current_paper['text'], entities
            )
            st.markdown(f'<div class="preview-box">{highlighted_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            
            methodology = analysis.get('methodology', [])
            if methodology:
                st.markdown('<div class="analysis-card"><h4>🔧 Extracted Methodology</h4>', unsafe_allow_html=True)
                for i, method in enumerate(methodology):
                    st.markdown(f"**{i+1}.** {method}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Methodology details not clearly identified.")
            
            st.markdown('<div class="analysis-card"><h4>💻 Technologies & Frameworks</h4>', unsafe_allow_html=True)
            entities = analysis.get('entities', {})
            technologies = entities.get('TECHNOLOGY', [])
            if technologies:
                for tech in technologies:
                    st.markdown(f"• {tech}")
            else:
                st.info("No specific technologies mentioned.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Chat Interface
    with col2:
        st.markdown('<h3 class="section-header">🤖 Research Assistant</h3>', unsafe_allow_html=True)
        
        if st.session_state.current_paper:
            st.markdown("""
            <div class="analysis-card">
                <h4>AI Assistant Status</h4>
                <p><strong>Status:</strong> <span style="color: #00b894;">Ready</span><br>
                <strong>Document Understanding:</strong> <span style="color: #00b894;">Active</span><br>
                <strong>Semantic Search:</strong> <span style="color: #00b894;">Enabled</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick question suggestions with beautiful cards
            st.markdown('<h4>💡 Suggested Questions</h4>', unsafe_allow_html=True)
            suggestions = [
                "What is the main contribution of this research?",
                "What methodologies are used?", 
                "What are the key findings?",
                "What technologies are mentioned?",
                "How does this compare to other work?"
            ]
            
            selected_suggestion = st.selectbox("Choose a quick question:", 
                                              [""] + suggestions, 
                                              key="suggestion_select")
            
            if selected_suggestion and st.button("Ask Selected Question", key="ask_suggestion", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": selected_suggestion})
                response = st.session_state.qa_processor.answer_question(selected_suggestion)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
        
        # Chat History Display
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div class="chat-message user"><strong>You:</strong> {message["content"]}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant"><strong>AI Assistant:</strong> {message["content"]}</div>', 
                              unsafe_allow_html=True)
        
        # Chat Input
        user_question = st.text_input("Ask anything about the document:", 
                                    key="user_question",
                                    placeholder="e.g., What is artificial intelligence?")
        
        col_send, col_clear = st.columns([3, 1])
        with col_send:
            send_button = st.button("📤 Send Question", type="primary", use_container_width=True)
        with col_clear:
            clear_button = st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True)
        
        if send_button and user_question:
            if st.session_state.current_paper:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                with st.spinner("🤔 Generating intelligent response..."):
                    response = st.session_state.qa_processor.answer_question(user_question)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
            else:
                st.warning("Please analyze a document first before asking questions.")
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown('<h2 class="section-header">📈 Analysis Dashboard</h2>', unsafe_allow_html=True)
        
        if st.session_state.processed_papers:
            st.markdown('<h4>📚 Document History</h4>', unsafe_allow_html=True)
            for i, paper in enumerate(reversed(st.session_state.processed_papers[-5:])):
                if st.button(f"📄 {paper['title'][:25]}...", key=f"history_{i}", use_container_width=True):
                    st.session_state.current_paper = paper
                    st.session_state.show_analysis = True
                    st.rerun()
        
        if st.session_state.current_paper:
            st.markdown("---")
            st.markdown('<h4>📊 Current Document Stats</h4>', unsafe_allow_html=True)
            
            analysis = st.session_state.current_paper.get('analysis', {})
            entities = analysis.get('entities', {})
            
            # Statistics
            total_entities = sum(len(v) for v in entities.values())
            word_count = len(st.session_state.current_paper['text'].split())
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="stat-box"><div class="stat-value">{:,}</div><div class="stat-label">Word Count</div></div>'.format(word_count), unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="stat-box"><div class="stat-value">{}</div><div class="stat-label">Total Entities</div></div>'.format(total_entities), unsafe_allow_html=True)
            
            st.markdown('<div class="stat-box"><div class="stat-value">{}</div><div class="stat-label">Entity Categories</div></div>'.format(len([k for k, v in entities.items() if v])), unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<h4>⚡ System Information</h4>', unsafe_allow_html=True)
        st.markdown("""
        <div class="analysis-card">
            <h4>🚀 Enhanced Features</h4>
            <span class="feature-badge">Document Analysis</span>
            <span class="feature-badge">Entity Extraction</span>
            <span class="feature-badge">Topic Modeling</span>
            <span class="feature-badge">Semantic QA</span>
            <span class="feature-badge">Multi-Strategy</span>
            <span class="feature-badge">Confidence Scoring</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear All Data", use_container_width=True):
            for key in ['processed_papers', 'current_paper', 'chat_history', 'qa_processor']:
                if key in st.session_state:
                    if key == 'qa_processor':
                        st.session_state[key] = IntelligentQASystem()
                    else:
                        st.session_state[key] = [] if key.endswith('papers') or key.endswith('history') else None
            st.success("✅ All data cleared!")
            st.rerun()

def highlight_entities_adaptive(text, entities):
    """Highlight entities in text with improved accuracy"""
    highlighted_text = text
    highlight_classes = {
        'RESEARCH_METHOD': 'highlight-method',
        'TECHNOLOGY': 'highlight-tech',
        'DATASET': 'highlight-dataset',
        'METRIC': 'highlight-metric'
    }
    
    # Sort entities by length (longest first) to avoid partial replacements
    all_entities = []
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            if entity and len(entity) > 2:
                all_entities.append((entity, highlight_classes.get(entity_type, 'highlight-method')))
    
    all_entities.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Replace entities with highlighted versions
    for entity, css_class in all_entities:
        # Use word boundaries to ensure exact matches
        pattern = rf'\b{re.escape(entity)}\b'
        highlighted_text = re.sub(
            pattern,
            f'<span class="highlight {css_class}" title="{entity}">{entity}</span>',
            highlighted_text,
            flags=re.IGNORECASE
        )
    
    return highlighted_text

if __name__ == "__main__":
    main()
