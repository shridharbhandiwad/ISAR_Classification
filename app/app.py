"""
ISAR Image Analysis - Professional Streamlit GUI Application
=============================================================

A comprehensive web application for training, evaluating, and deploying
deep learning models for ISAR image classification of automotive targets.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import io
import tempfile

# Import project modules
from src.utils.config import load_config, save_config
from src.utils.helpers import get_device, set_seed
from src.models import create_model, get_available_models
from src.data import ISARDataset, create_data_loaders, generate_synthetic_isar_data
from src.training import Trainer, TrainingConfig, MetricsTracker
from src.evaluation import Evaluator, GradCAMExplainer
from src.visualization import (
    plot_confusion_matrix_plotly, plot_roc_curves_plotly,
    plot_training_history_plotly, plot_feature_embeddings_plotly
)
from src.reports import ReportGenerator


# Page configuration
st.set_page_config(
    page_title="ISAR Image Analysis",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
    }
    .success-box {
        background: #ecfdf5;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        color: #065f46;
    }
    .warning-box {
        background: #fffbeb;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        color: #92400e;
    }
    .info-box {
        background: #eff6ff;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = None
    if 'training_history' not in st.session_state:
        st.session_state.training_history = None
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = None
    if 'config' not in st.session_state:
        try:
            st.session_state.config = load_config('config/config.yaml')
        except:
            st.session_state.config = {}


def main():
    """Main application entry point."""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📡 ISAR Image Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning Platform for Inverse Synthetic Aperture Radar Image Classification</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "📊 Data Explorer", "🧠 Model Training", 
         "📈 Evaluation", "🔮 Inference", "📋 Reports"]
    )
    
    # Device info in sidebar
    device = get_device()
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Device:** {device}")
    if torch.cuda.is_available():
        st.sidebar.markdown(f"**GPU:** {torch.cuda.get_device_name(0)}")
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "🧠 Model Training":
        show_training_page()
    elif page == "📈 Evaluation":
        show_evaluation_page()
    elif page == "🔮 Inference":
        show_inference_page()
    elif page == "📋 Reports":
        show_reports_page()


def show_home_page():
    """Display home page with project overview."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 About ISAR
        
        **Inverse Synthetic Aperture Radar (ISAR)** imaging is a radar 
        technique used to generate high-resolution 2D images of targets. 
        It's widely used in:
        - Automotive radar systems
        - Maritime surveillance
        - Air traffic control
        - Military applications
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Our Models
        
        We support multiple deep learning architectures:
        - **Custom CNN** - Optimized for ISAR images
        - **ResNet** - 18, 34, 50 variants
        - **EfficientNet** - B0, B1, B2 variants
        - **Vision Transformer** - Tiny, Small, Base
        """)
    
    with col3:
        st.markdown("""
        ### 📈 Features
        
        - Real-time training visualization
        - Interactive model evaluation
        - Grad-CAM explanations
        - Automated report generation
        - Model comparison tools
        """)
    
    st.markdown("---")
    
    # Quick start guide
    st.subheader("🚀 Quick Start Guide")
    
    st.markdown("""
    1. **Data Explorer**: Upload your ISAR dataset or generate synthetic data
    2. **Model Training**: Configure and train deep learning models
    3. **Evaluation**: Analyze model performance with detailed metrics
    4. **Inference**: Make predictions on new images
    5. **Reports**: Generate professional PDF/HTML reports
    """)
    
    # Model status
    st.subheader("📊 Current Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "✅ Loaded" if st.session_state.model else "❌ Not Loaded"
        st.metric("Model", status)
    
    with col2:
        classes = len(st.session_state.class_names) if st.session_state.class_names else 0
        st.metric("Classes", classes)
    
    with col3:
        trained = "✅ Yes" if st.session_state.training_history else "❌ No"
        st.metric("Trained", trained)
    
    with col4:
        evaluated = "✅ Yes" if st.session_state.eval_results else "❌ No"
        st.metric("Evaluated", evaluated)


def show_data_explorer():
    """Display data exploration page."""
    st.header("📊 Data Explorer")
    
    tab1, tab2, tab3 = st.tabs(["📁 Load Dataset", "🎲 Generate Synthetic", "📈 Visualize"])
    
    with tab1:
        st.subheader("Load Dataset")
        
        data_dir = st.text_input(
            "Dataset Directory",
            value="data/raw",
            help="Path to directory containing class subdirectories"
        )
        
        if st.button("Load Dataset", key="load_data"):
            with st.spinner("Loading dataset..."):
                try:
                    dataset = ISARDataset(data_dir=data_dir)
                    st.session_state.class_names = dataset.classes
                    
                    st.success(f"✅ Loaded {len(dataset)} samples from {len(dataset.classes)} classes")
                    
                    # Show distribution
                    distribution = dataset.get_class_distribution()
                    
                    fig = px.bar(
                        x=list(distribution.keys()),
                        y=list(distribution.values()),
                        title="Class Distribution",
                        labels={'x': 'Class', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {str(e)}")
    
    with tab2:
        st.subheader("Generate Synthetic ISAR Data")
        
        st.info("""
        Generate synthetic ISAR-like images for testing and demonstration.
        This creates realistic radar signatures of automotive targets.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            output_dir = st.text_input("Output Directory", value="data/raw")
            num_samples = st.slider("Samples per Class", 50, 500, 200)
        
        with col2:
            image_size = st.selectbox("Image Size", [64, 128, 256], index=1)
            noise_level = st.slider("Noise Level", 0.0, 0.3, 0.1)
        
        if st.button("Generate Dataset", key="generate_data"):
            with st.spinner("Generating synthetic data..."):
                try:
                    output_path, stats = generate_synthetic_isar_data(
                        output_dir=output_dir,
                        num_samples_per_class=num_samples,
                        image_size=image_size,
                        noise_level=noise_level
                    )
                    
                    st.session_state.class_names = stats['classes']
                    
                    st.success(f"✅ Generated {stats['total_samples']} samples in {output_path}")
                    
                    # Show stats
                    st.json(stats)
                    
                except Exception as e:
                    st.error(f"❌ Error generating data: {str(e)}")
    
    with tab3:
        st.subheader("Data Visualization")
        
        if st.session_state.class_names:
            st.write(f"**Classes:** {', '.join(st.session_state.class_names)}")
        else:
            st.warning("Load a dataset first to visualize samples.")


def show_training_page():
    """Display model training page."""
    st.header("🧠 Model Training")
    
    # Configuration section
    with st.expander("⚙️ Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Model")
            architecture = st.selectbox(
                "Architecture",
                get_available_models(),
                index=2  # resnet18
            )
            num_classes = st.number_input("Number of Classes", 2, 20, 5)
            pretrained = st.checkbox("Use Pretrained Weights", value=True)
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.3)
        
        with col2:
            st.subheader("Training")
            epochs = st.number_input("Epochs", 10, 500, 100)
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64, 128], index=2)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001
            )
            optimizer = st.selectbox("Optimizer", ['adamw', 'adam', 'sgd'])
        
        with col3:
            st.subheader("Data")
            data_dir = st.text_input("Data Directory", value="data/raw")
            image_size = st.selectbox("Image Size", [64, 128, 256], index=1)
            train_ratio = st.slider("Train Ratio", 0.5, 0.9, 0.7)
            
            scheduler = st.selectbox(
                "LR Scheduler",
                ['cosine', 'step', 'plateau', 'none']
            )
    
    # Training button
    st.markdown("---")
    
    if st.button("🚀 Start Training", type="primary"):
        # Validate data directory
        if not os.path.exists(data_dir):
            st.error(f"❌ Data directory not found: {data_dir}")
            return
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()
        
        try:
            # Create data loaders
            status_text.text("Loading data...")
            train_loader, val_loader, test_loader, class_names = create_data_loaders(
                data_dir=data_dir,
                batch_size=batch_size,
                image_size=image_size,
                train_ratio=train_ratio,
                val_ratio=(1 - train_ratio) / 2,
                test_ratio=(1 - train_ratio) / 2
            )
            
            st.session_state.class_names = class_names
            
            # Create model
            status_text.text("Creating model...")
            model = create_model(
                architecture=architecture,
                num_classes=num_classes,
                in_channels=1,
                pretrained=pretrained,
                dropout=dropout
            )
            
            # Training config
            config = TrainingConfig(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                optimizer=optimizer,
                scheduler_type=scheduler,
                early_stopping_enabled=True,
                early_stopping_patience=15
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                class_names=class_names
            )
            
            # Training loop with progress updates
            status_text.text("Training...")
            
            history = []
            best_val_acc = 0
            
            for epoch in range(epochs):
                # Train epoch
                train_metrics = trainer._train_epoch()
                val_metrics = trainer._validate_epoch()
                
                # Update scheduler
                if trainer.scheduler is not None:
                    if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        trainer.scheduler.step(val_metrics['loss'])
                    else:
                        trainer.scheduler.step()
                
                # Record history
                epoch_data = {
                    'epoch': epoch,
                    'train': train_metrics,
                    'val': val_metrics
                }
                history.append(epoch_data)
                
                # Update best model
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    trainer._save_checkpoint(True, val_metrics)
                
                # Update progress
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )
                
                # Update metrics display
                with metrics_container.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Train Loss", f"{train_metrics['loss']:.4f}")
                    col2.metric("Val Loss", f"{val_metrics['loss']:.4f}")
                    col3.metric("Train Acc", f"{train_metrics['accuracy']:.4f}")
                    col4.metric("Val Acc", f"{val_metrics['accuracy']:.4f}")
                
                # Check early stopping
                if trainer.early_stopping and trainer.early_stopping(val_metrics['loss']):
                    status_text.text(f"Early stopping at epoch {epoch + 1}")
                    break
                
                trainer.current_epoch = epoch
            
            # Save results
            st.session_state.model = model
            st.session_state.model_name = architecture
            st.session_state.training_history = history
            
            st.success(f"✅ Training completed! Best validation accuracy: {best_val_acc:.4f}")
            
            # Show training curves
            if history:
                fig = plot_training_history_plotly(history)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Training error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def show_evaluation_page():
    """Display model evaluation page."""
    st.header("📈 Model Evaluation")
    
    if st.session_state.model is None:
        st.warning("⚠️ No model loaded. Please train or load a model first.")
        
        # Option to load checkpoint
        checkpoint_path = st.text_input("Or load from checkpoint", value="checkpoints/best_model.pt")
        
        if st.button("Load Checkpoint"):
            try:
                # Create model
                architecture = st.selectbox("Architecture", get_available_models())
                num_classes = st.number_input("Number of Classes", 2, 20, 5)
                
                model = create_model(architecture, num_classes=num_classes, pretrained=False)
                
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                st.session_state.model = model
                st.session_state.model_name = architecture
                st.success("✅ Model loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error loading checkpoint: {str(e)}")
        return
    
    # Evaluation settings
    col1, col2 = st.columns(2)
    
    with col1:
        data_dir = st.text_input("Test Data Directory", value="data/raw")
    
    with col2:
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
    
    if st.button("🔍 Evaluate Model", type="primary"):
        with st.spinner("Evaluating model..."):
            try:
                # Create test loader
                _, _, test_loader, class_names = create_data_loaders(
                    data_dir=data_dir,
                    batch_size=batch_size,
                    train_ratio=0.7,
                    val_ratio=0.15,
                    test_ratio=0.15
                )
                
                # Create evaluator
                evaluator = Evaluator(
                    model=st.session_state.model,
                    class_names=class_names
                )
                
                # Run evaluation
                results = evaluator.evaluate(test_loader)
                st.session_state.eval_results = results
                st.session_state.class_names = class_names
                
                # Display results
                st.subheader("📊 Results")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{results['accuracy']:.2%}")
                col2.metric("Precision", f"{results['metrics']['precision']:.2%}")
                col3.metric("Recall", f"{results['metrics']['recall']:.2%}")
                col4.metric("F1 Score", f"{results['metrics']['f1']:.2%}")
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm = np.array(results['confusion_matrix'])
                fig = plot_confusion_matrix_plotly(cm, class_names)
                st.plotly_chart(fig, use_container_width=True)
                
                # Per-class metrics
                st.subheader("Per-Class Performance")
                per_class_df = []
                for cls, metrics in results['per_class_metrics'].items():
                    per_class_df.append({
                        'Class': cls,
                        'Precision': f"{metrics['precision']:.2%}",
                        'Recall': f"{metrics['recall']:.2%}",
                        'F1 Score': f"{metrics['f1']:.2%}"
                    })
                st.table(per_class_df)
                
                # Classification report
                with st.expander("📋 Full Classification Report"):
                    st.text(results['classification_report'])
                
            except Exception as e:
                st.error(f"❌ Evaluation error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def show_inference_page():
    """Display inference page for making predictions."""
    st.header("🔮 Model Inference")
    
    if st.session_state.model is None:
        st.warning("⚠️ No model loaded. Please train or load a model first.")
        return
    
    st.info("Upload an ISAR image to get predictions from the trained model.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload ISAR Image",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff']
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            image = Image.open(uploaded_file).convert('L')
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("Prediction")
            
            # Preprocess image
            import torchvision.transforms as T
            
            transform = T.Compose([
                T.Resize((128, 128)),
                T.ToTensor()
            ])
            
            img_tensor = transform(image).unsqueeze(0)
            
            # Make prediction
            st.session_state.model.eval()
            device = get_device()
            img_tensor = img_tensor.to(device)
            st.session_state.model.to(device)
            
            with torch.no_grad():
                output = st.session_state.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_class = output.argmax(dim=1).item()
            
            # Get class name
            if st.session_state.class_names:
                class_name = st.session_state.class_names[predicted_class]
            else:
                class_name = f"Class {predicted_class}"
            
            confidence = probabilities[predicted_class].item()
            
            # Display prediction
            st.metric("Predicted Class", class_name)
            st.metric("Confidence", f"{confidence:.2%}")
            
            # Show all probabilities
            st.subheader("Class Probabilities")
            
            if st.session_state.class_names:
                prob_data = {
                    'Class': st.session_state.class_names,
                    'Probability': probabilities.cpu().numpy()
                }
            else:
                prob_data = {
                    'Class': [f"Class {i}" for i in range(len(probabilities))],
                    'Probability': probabilities.cpu().numpy()
                }
            
            fig = px.bar(
                x=prob_data['Class'],
                y=prob_data['Probability'],
                title="Prediction Probabilities"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Grad-CAM visualization
    st.markdown("---")
    st.subheader("🔍 Model Explanation (Grad-CAM)")
    
    if uploaded_file is not None and st.button("Generate Grad-CAM"):
        with st.spinner("Generating Grad-CAM visualization..."):
            try:
                # Create explainer
                explainer = GradCAMExplainer(st.session_state.model)
                
                # Generate CAM
                cam, pred_class, conf = explainer.generate_cam(img_tensor)
                
                # Create overlay
                img_np = np.array(image)
                overlay = explainer.overlay_cam(img_np, cam)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(img_np, caption="Original", use_container_width=True)
                
                with col2:
                    st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error generating Grad-CAM: {str(e)}")


def show_reports_page():
    """Display reports generation page."""
    st.header("📋 Report Generation")
    
    st.info("Generate professional reports of your training and evaluation results.")
    
    # Report type selection
    report_type = st.selectbox(
        "Report Type",
        ["Training Report", "Evaluation Report", "Model Comparison"]
    )
    
    # Format selection
    col1, col2 = st.columns(2)
    
    with col1:
        report_format = st.selectbox("Format", ["HTML", "PDF", "Both"])
    
    with col2:
        output_dir = st.text_input("Output Directory", value="reports")
    
    # Generate report button
    if st.button("📄 Generate Report", type="primary"):
        if report_type == "Training Report" and st.session_state.training_history is None:
            st.warning("⚠️ No training history available. Train a model first.")
            return
        
        if report_type == "Evaluation Report" and st.session_state.eval_results is None:
            st.warning("⚠️ No evaluation results available. Evaluate a model first.")
            return
        
        with st.spinner("Generating report..."):
            try:
                generator = ReportGenerator(output_dir=output_dir)
                
                if report_type == "Training Report":
                    path = generator.generate_training_report(
                        training_results={
                            'history': st.session_state.training_history,
                            'best_val_acc': max(h['val']['accuracy'] for h in st.session_state.training_history),
                            'best_val_loss': min(h['val']['loss'] for h in st.session_state.training_history),
                            'epochs_trained': len(st.session_state.training_history),
                            'total_time': 0
                        },
                        config=st.session_state.config,
                        model_name=st.session_state.model_name or "Unknown",
                        format=report_format.lower()
                    )
                    
                elif report_type == "Evaluation Report":
                    path = generator.generate_evaluation_report(
                        eval_results=st.session_state.eval_results,
                        model_name=st.session_state.model_name or "Unknown",
                        class_names=st.session_state.class_names or [],
                        format=report_format.lower()
                    )
                
                st.success(f"✅ Report generated: {path}")
                
                # Provide download link for HTML
                if isinstance(path, list):
                    for p in path:
                        if p.endswith('.html'):
                            with open(p, 'r') as f:
                                st.download_button(
                                    "📥 Download HTML Report",
                                    f.read(),
                                    file_name=os.path.basename(p),
                                    mime="text/html"
                                )
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")


if __name__ == "__main__":
    main()
