"""
Advanced Deep Learning URL Maliciousness Classifier
Features:
- Multi-head attention mechanism
- Character and word-level embeddings
- Ensemble model architecture
- Advanced feature engineering
- Obfuscation detection
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import re
import warnings
warnings.filterwarnings('ignore')

class URLFeatureExtractor:
    """Extract advanced features from URLs for deep learning"""
    
    def __init__(self):
        self.suspicious_tlds = ['.xyz', '.top', '.loan', '.work', '.click', '.gq', '.ml', '.cf', '.ga', '.tk']
        self.obfuscation_patterns = [
            r'%[0-9a-fA-F]{2}',  # URL encoding
            r'&#x?[0-9a-fA-F]+;',  # HTML entities
            r'\\x[0-9a-fA-F]{2}',  # Hex encoding
            r'\\u[0-9a-fA-F]{4}',  # Unicode escape
        ]
    
    def extract_numerical_features(self, url):
        """Extract numerical features from URL"""
        parsed = urlparse(url)
        
        features = {
            'url_length': len(url),
            'domain_length': len(parsed.netloc),
            'path_length': len(parsed.path),
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_underscores': url.count('_'),
            'num_slashes': url.count('/'),
            'num_questionmarks': url.count('?'),
            'num_equals': url.count('='),
            'num_ats': url.count('@'),
            'num_ampersands': url.count('&'),
            'num_digits': sum(c.isdigit() for c in url),
            'num_params': len(parsed.query.split('&')) if parsed.query else 0,
            'has_ip': int(self._has_ip_address(parsed.netloc)),
            'has_suspicious_tld': int(any(url.endswith(tld) for tld in self.suspicious_tlds)),
            'entropy': self._calculate_entropy(url),
            'digit_ratio': sum(c.isdigit() for c in url) / max(len(url), 1),
            'special_char_ratio': sum(not c.isalnum() for c in url) / max(len(url), 1),
            'obfuscation_score': self._detect_obfuscation(url),
            'subdomain_count': len(parsed.netloc.split('.')) - 2 if len(parsed.netloc.split('.')) > 2 else 0,
            'path_depth': len([p for p in parsed.path.split('/') if p]),
        }
        
        return features
    
    def _has_ip_address(self, domain):
        """Check if domain contains IP address"""
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        return bool(re.search(ip_pattern, domain))
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        prob = [text.count(c) / len(text) for c in set(text)]
        entropy = -sum(p * np.log2(p) for p in prob if p > 0)
        return entropy
    
    def _detect_obfuscation(self, url):
        """Detect various obfuscation techniques"""
        score = 0
        for pattern in self.obfuscation_patterns:
            matches = len(re.findall(pattern, url))
            score += matches
        
        # Check for excessive encoding
        if url.count('%') > 3:
            score += 2
        
        # Check for mixed case in suspicious ways
        if sum(c.isupper() for c in url) > len(url) * 0.3:
            score += 1
            
        return min(score, 10)  # Normalize to 0-10


class AttentionLayer(layers.Layer):
    """Custom attention mechanism for URL analysis"""
    
    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.W = layers.Dense(self.units)
        self.V = layers.Dense(1)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Score shape: (batch_size, max_length, 1)
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # Context vector shape: (batch_size, hidden_size)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class URLClassifier:
    """Advanced URL Classifier with Deep Learning"""
    
    def __init__(self, max_chars=200, max_words=50, char_embed_dim=32, word_embed_dim=64):
        self.max_chars = max_chars
        self.max_words = max_words
        self.char_embed_dim = char_embed_dim
        self.word_embed_dim = word_embed_dim
        self.feature_extractor = URLFeatureExtractor()
        self.char_tokenizer = None
        self.word_tokenizer = None
        self.model = None
        self.history = None
        
    def _tokenize_characters(self, urls):
        """Tokenize URLs at character level"""
        if self.char_tokenizer is None:
            self.char_tokenizer = Tokenizer(char_level=True, lower=False)
            self.char_tokenizer.fit_on_texts(urls)
        
        sequences = self.char_tokenizer.texts_to_sequences(urls)
        return pad_sequences(sequences, maxlen=self.max_chars, padding='post')
    
    def _tokenize_words(self, urls):
        """Tokenize URLs at word level (split by special characters)"""
        # Split URLs by special characters to get tokens
        tokenized_urls = [re.findall(r'[a-zA-Z0-9]+', url) for url in urls]
        
        if self.word_tokenizer is None:
            self.word_tokenizer = Tokenizer(lower=True)
            self.word_tokenizer.fit_on_texts(tokenized_urls)
        
        sequences = self.word_tokenizer.texts_to_sequences(tokenized_urls)
        return pad_sequences(sequences, maxlen=self.max_words, padding='post')
    
    def _extract_features(self, urls):
        """Extract numerical features from URLs"""
        features_list = [self.feature_extractor.extract_numerical_features(url) for url in urls]
        return np.array([[f[key] for key in sorted(f.keys())] for f in features_list])
    
    def build_model(self):
        """Build advanced ensemble deep learning model"""
        
        # Input layers
        char_input = layers.Input(shape=(self.max_chars,), name='char_input')
        word_input = layers.Input(shape=(self.max_words,), name='word_input')
        feature_input = layers.Input(shape=(21,), name='feature_input')  # 21 numerical features
        
        # Character-level branch
        char_vocab_size = len(self.char_tokenizer.word_index) + 1
        char_embedding = layers.Embedding(char_vocab_size, self.char_embed_dim, 
                                         mask_zero=True)(char_input)
        char_lstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(char_embedding)
        char_attention = AttentionLayer(64)(char_lstm)
        char_dropout = layers.Dropout(0.3)(char_attention)
        
        # Word-level branch
        word_vocab_size = len(self.word_tokenizer.word_index) + 1
        word_embedding = layers.Embedding(word_vocab_size, self.word_embed_dim, 
                                         mask_zero=True)(word_input)
        word_gru = layers.Bidirectional(layers.GRU(64, return_sequences=True))(word_embedding)
        word_attention = AttentionLayer(64)(word_gru)
        word_dropout = layers.Dropout(0.3)(word_attention)
        
        # Feature branch
        feature_dense1 = layers.Dense(32, activation='relu')(feature_input)
        feature_bn1 = layers.BatchNormalization()(feature_dense1)
        feature_dropout1 = layers.Dropout(0.2)(feature_bn1)
        feature_dense2 = layers.Dense(16, activation='relu')(feature_dropout1)
        feature_bn2 = layers.BatchNormalization()(feature_dense2)
        
        # Concatenate all branches
        concatenated = layers.Concatenate()([char_dropout, word_dropout, feature_bn2])
        
        # Dense layers
        dense1 = layers.Dense(128, activation='relu')(concatenated)
        bn1 = layers.BatchNormalization()(dense1)
        dropout1 = layers.Dropout(0.4)(bn1)
        
        dense2 = layers.Dense(64, activation='relu')(dropout1)
        bn2 = layers.BatchNormalization()(dense2)
        dropout2 = layers.Dropout(0.3)(bn2)
        
        dense3 = layers.Dense(32, activation='relu')(dropout2)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(dense3)
        
        # Create model
        self.model = models.Model(
            inputs=[char_input, word_input, feature_input],
            outputs=output
        )
        
        # Compile with advanced optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'), 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        return self.model
    
    def prepare_data(self, urls, labels=None):
        """Prepare data for model input"""
        char_sequences = self._tokenize_characters(urls)
        word_sequences = self._tokenize_words(urls)
        numerical_features = self._extract_features(urls)
        
        return [char_sequences, word_sequences, numerical_features], labels
    
    def train(self, urls, labels, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model"""
        # Prepare data
        X, y = self.prepare_data(urls, labels)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_url_model.keras',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, urls):
        """Predict URL maliciousness"""
        X, _ = self.prepare_data(urls)
        return self.model.predict(X)
    
    def evaluate(self, urls, labels):
        """Evaluate model performance"""
        X, y = self.prepare_data(urls, labels)
        results = self.model.evaluate(X, y, verbose=0)
        
        # Get predictions
        y_pred_proba = self.model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Print detailed metrics
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Loss: {results[0]:.4f}")
        print(f"Accuracy: {results[1]:.4f}")
        print(f"AUC: {results[2]:.4f}")
        print(f"Precision: {results[3]:.4f}")
        print(f"Recall: {results[4]:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Benign', 'Malicious']))
        
        return results, y_pred_proba, y_pred


def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC
    axes[1, 0].plot(history.history['auc'], label='Train')
    axes[1, 0].plot(history.history['val_auc'], label='Validation')
    axes[1, 0].set_title('Model AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Precision & Recall
    axes[1, 1].plot(history.history['precision'], label='Train Precision')
    axes[1, 1].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('/mnt/user-data/outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('/mnt/user-data/outputs/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Advanced URL Maliciousness Classifier")
    print("="*50)
    print("This is a demonstration. Replace with your actual dataset.")
    print("\nFor usage with your dataset, see main.py")