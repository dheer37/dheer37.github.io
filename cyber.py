
import os
import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ========== CORE DETECTION MODULES ==========

class VoiceDeepfakeDetector:
    """Detects AI-generated voice deepfakes"""
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.load_model()
        
    def load_model(self):
        """Load pre-trained model or train locally"""
        model_path = 'models/voice_detector.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = self.train_local_model()
            
    def train_local_model(self):
        """Train voice detection model with local data"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Generate synthetic dataset
        real_voices, fake_voices = self.generate_dataset()
        
        # Extract features
        X, y = [], []
        for voice in real_voices:
            features = self.extract_audio_features(voice)
            X.append(features)
            y.append(0)  # Real
        
        for voice in fake_voices:
            features = self.extract_audio_features(voice)
            X.append(features)
            y.append(1)  # Fake
            
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save for future use
        with open('models/voice_detector.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        return model
    
    def analyze(self, audio_path: str) -> Dict:
        """Analyze audio file for deepfake indicators"""
        features = self.extract_audio_features(audio_path)
        prediction = self.model.predict([features])[0]
        probability = self.model.predict_proba([features])[0]
        
        return {
            'type': 'voice',
            'is_deepfake': bool(prediction),
            'confidence': float(probability[prediction]),
            'risk_level': self.calculate_risk_level(probability[prediction]),
            'indicators': self.get_detection_indicators(features),
            'recommendations': self.get_recommendations(prediction)
        }


class VideoDeepfakeDetector:
    """Detects video deepfakes using frame-by-frame analysis"""
    
    def __init__(self):
        self.models = {
            'blink': self.load_blink_model(),
            'lighting': self.load_lighting_model(),
            'background': self.load_background_model()
        }
        
    def analyze(self, video_path: str) -> Dict:
        """Analyze video file for deepfake indicators"""
        frames = self.extract_key_frames(video_path)
        
        analyses = []
        for frame in frames:
            analysis = self.analyze_frame(frame)
            analyses.append(analysis)
        
        # Aggregate results
        is_deepfake = self.aggregate_predictions(analyses)
        confidence = self.calculate_confidence(analyses)
        
        return {
            'type': 'video',
            'is_deepfake': is_deepfake,
            'confidence': confidence,
            'total_frames_analyzed': len(frames),
            'anomalous_frames': self.count_anomalies(analyses),
            'temporal_inconsistencies': self.find_temporal_issues(analyses),
            'visual_artifacts': self.detect_artifacts(frames)
        }


class ImageDeepfakeDetector:
    """Detects image deepfakes and manipulated photos"""
    
    def __init__(self):
        self.model = self.load_cnn_model()
        
    def analyze(self, image_path: str) -> Dict:
        """Analyze image for manipulation signs"""
        image = self.preprocess_image(image_path)
        features = self.extract_image_features(image)
        
        prediction = self.model.predict([features])[0]
        probability = self.model.predict_proba([features])[0]
        
        # Detect specific manipulation types
        manipulations = self.detect_manipulation_types(image)
        
        return {
            'type': 'image',
            'is_deepfake': bool(prediction),
            'confidence': float(probability[prediction]),
            'manipulations_detected': manipulations,
            'ela_score': self.error_level_analysis(image),
            'noise_inconsistency': self.check_noise_patterns(image),
            'face_landmark_analysis': self.analyze_face_landmarks(image)
        }


# ========== UNIFIED SECURITY SUITE ==========

@dataclass
class AnalysisResult:
    """Unified result structure for all detection types"""
    file_type: str
    is_deepfake: bool
    confidence: float
    risk_level: str  # low, medium, high, critical
    indicators: List[str]
    recommendations: List[str]
    metadata: Dict
    timestamp: str
    analysis_duration: float


class DeepfakeSecuritySuite:
    """
    MASTER CLASS: Combines all detection modules
    Single interface for voice, video, and image deepfake detection
    """
    
    def __init__(self, data_dir: str = "deepfake_data"):
        print(" Initializing Deepfake Security Suite...")
        print(" All processing will be done locally on this machine")
        
        # Initialize all detectors
        self.voice_detector = VoiceDeepfakeDetector()
        self.video_detector = VideoDeepfakeDetector()
        self.image_detector = ImageDeepfakeDetector()
        
        # Setup directories
        self.data_dir = Path(data_dir)
        self.setup_directories()
        
        # Performance metrics
        self.analyses_history = []
        
        print(" Suite initialized successfully!")
        print(f" Available detectors: Voice | Video | Image")
    
    def setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            'models',
            'data/raw',
            'data/processed', 
            'data/synthetic',
            'results',
            'logs',
            'exports'
        ]
        
        for dir_path in directories:
            (self.data_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def analyze_file(self, file_path: str) -> AnalysisResult:
        """
        Universal analysis function - automatically detects file type
        and routes to appropriate detector
        """
        print(f"\n Analyzing: {file_path}")
        
        start_time = datetime.now()
        
        # Determine file type
        file_ext = Path(file_path).suffix.lower()
        file_type = self.determine_file_type(file_ext)
        
        if file_type == 'unknown':
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Route to appropriate detector
        if file_type == 'audio':
            detector_result = self.voice_detector.analyze(file_path)
        elif file_type == 'video':
            detector_result = self.video_detector.analyze(file_path)
        elif file_type == 'image':
            detector_result = self.image_detector.analyze(file_path)
        
        # Calculate analysis duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Create unified result
        result = AnalysisResult(
            file_type=file_type,
            is_deepfake=detector_result['is_deepfake'],
            confidence=detector_result['confidence'],
            risk_level=self.calculate_risk_level(
                detector_result['confidence'],
                detector_result.get('is_deepfake', False)
            ),
            indicators=detector_result.get('indicators', []),
            recommendations=detector_result.get('recommendations', []),
            metadata={
                'file_size': os.path.getsize(file_path),
                'file_type': file_ext,
                'detector_used': file_type,
                'analysis_method': self.get_analysis_method(file_type)
            },
            timestamp=datetime.now().isoformat(),
            analysis_duration=duration
        )
        
        # Store in history
        self.analyses_history.append(result)
        
        return result
    
    def determine_file_type(self, extension: str) -> str:
        """Determine file type from extension"""
        audio_ext = {'.wav', '.mp3', '.m4a', '.aac', '.flac'}
        video_ext = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        image_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        if extension in audio_ext:
            return 'audio'
        elif extension in video_ext:
            return 'video'
        elif extension in image_ext:
            return 'image'
        else:
            return 'unknown'
    
    def calculate_risk_level(self, confidence: float, is_deepfake: bool) -> str:
        """Calculate risk level based on confidence score"""
        if not is_deepfake:
            return 'low'
        
        if confidence >= 0.9:
            return 'critical'
        elif confidence >= 0.7:
            return 'high'
        elif confidence >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def batch_analyze(self, directory_path: str) -> Dict:
        """Analyze all supported files in a directory"""
        directory = Path(directory_path)
        results = {}
        
        print(f"\n Batch analyzing directory: {directory}")
        
        # Find all supported files
        supported_ext = {'.wav', '.mp3', '.mp4', '.avi', '.jpg', '.png'}
        files = []
        for ext in supported_ext:
            files.extend(directory.glob(f'*{ext}'))
            files.extend(directory.glob(f'*{ext.upper()}'))
        
        print(f"📊 Found {len(files)} files to analyze")
        
        for i, file_path in enumerate(files, 1):
            print(f"  [{i}/{len(files)}] Analyzing {file_path.name}")
            try:
                result = self.analyze_file(str(file_path))
                results[str(file_path)] = result
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results[str(file_path)] = {'error': str(e)}
        
        return results
    
    def generate_report(self, results: List[AnalysisResult]) -> Dict:
        """Generate comprehensive report from analysis results"""
        total = len(results)
        deepfakes = sum(1 for r in results if r.is_deepfake)
        
        report = {
            'summary': {
                'total_files': total,
                'deepfakes_detected': deepfakes,
                'deepfake_rate': deepfakes / total if total > 0 else 0,
                'analysis_timestamp': datetime.now().isoformat(),
                'suite_version': '1.0.0'
            },
            'by_file_type': self.aggregate_by_file_type(results),
            'risk_distribution': self.calculate_risk_distribution(results),
            'confidence_statistics': self.calculate_confidence_stats(results),
            'detailed_results': [
                {
                    'file_type': r.file_type,
                    'is_deepfake': r.is_deepfake,
                    'confidence': r.confidence,
                    'risk_level': r.risk_level,
                    'timestamp': r.timestamp
                }
                for r in results
            ]
        }
        
        return report
    
    def real_time_monitor(self, source: str = 'webcam', duration: int = 60):
        """
        Real-time monitoring for webcam or microphone
        Detects deepfakes as they occur
        """
        print(f"\n🎥 Starting real-time {source} monitoring...")
        print("Press Ctrl+C to stop")
        
        if source == 'webcam':
            self.monitor_webcam(duration)
        elif source == 'microphone':
            self.monitor_microphone(duration)
        else:
            print(f"Unknown source: {source}")
    
    def monitor_webcam(self, duration: int):
        """Monitor webcam feed for deepfakes"""
        import cv2
        import time
        
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        frame_count = 0
        
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze every 10th frame for performance
            if frame_count % 10 == 0:
                # Save frame temporarily
                temp_file = f"temp/frame_{frame_count}.jpg"
                cv2.imwrite(temp_file, frame)
                
                # Analyze
                result = self.image_detector.analyze(temp_file)
                
                if result['is_deepfake']:
                    print(f"🚨 DEEPFAKE DETECTED in frame {frame_count}")
                    print(f"   Confidence: {result['confidence']:.2f}")
            
            # Display feed
            cv2.imshow('Deepfake Monitor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def export_results(self, results: List[AnalysisResult], format: str = 'json'):
        """Export analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            filename = f"exports/deepfake_analysis_{timestamp}.json"
            data = [self.result_to_dict(r) for r in results]
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Results exported to {filename}")
        
        elif format == 'csv':
            filename = f"exports/deepfake_analysis_{timestamp}.csv"
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['File Type', 'Is Deepfake', 'Confidence', 
                               'Risk Level', 'Timestamp', 'Duration'])
                for r in results:
                    writer.writerow([
                        r.file_type, r.is_deepfake, r.confidence,
                        r.risk_level, r.timestamp, r.analysis_duration
                    ])
            print(f"✅ Results exported to {filename}")
    
    def result_to_dict(self, result: AnalysisResult) -> Dict:
        """Convert AnalysisResult to dictionary"""
        return {
            'file_type': result.file_type,
            'is_deepfake': result.is_deepfake,
            'confidence': result.confidence,
            'risk_level': result.risk_level,
            'indicators': result.indicators,
            'recommendations': result.recommendations,
            'metadata': result.metadata,
            'timestamp': result.timestamp,
            'analysis_duration': result.analysis_duration
        }


# ========== TRAINING MODULE ==========

class DeepfakeModelTrainer:
    """
    Unified trainer for all three detection models
    Creates synthetic data and trains models locally
    """
    
    def __init__(self, suite: DeepfakeSecuritySuite):
        self.suite = suite
        self.training_data = {}
        
    def generate_training_data(self, samples_per_type: int = 100):
        """Generate synthetic training data for all three types"""
        print("\n🎯 Generating training data...")
        
        # Generate voice data
        print("  🔊 Generating voice samples...")
        self.training_data['voice'] = self.generate_voice_data(samples_per_type)
        
        # Generate video data
        print("  🎬 Generating video samples...")
        self.training_data['video'] = self.generate_video_data(samples_per_type)
        
        # Generate image data
        print("  📸 Generating image samples...")
        self.training_data['image'] = self.generate_image_data(samples_per_type)
        
        print(f"✅ Generated {samples_per_type * 2} samples per type")
    
    def train_all_models(self):
        """Train all three detection models"""
        print("\n🎯 Training all models...")
        
        # Train voice model
        print("  🔊 Training voice detection model...")
        self.train_voice_model()
        
        # Train video model
        print("  🎬 Training video detection model...")
        self.train_video_model()
        
        # Train image model
        print("  📸 Training image detection model...")
        self.train_image_model()
        
        print("✅ All models trained successfully!")
    
    def evaluate_models(self, test_size: float = 0.2):
        """Evaluate all models on test data"""
        print("\n📊 Evaluating models...")
        
        results = {}
        
        # Evaluate voice model
        voice_acc = self.evaluate_voice_model(test_size)
        results['voice'] = voice_acc
        
        # Evaluate video model
        video_acc = self.evaluate_video_model(test_size)
        results['video'] = video_acc
        
        # Evaluate image model
        image_acc = self.evaluate_image_model(test_size)
        results['image'] = image_acc
        
        print("\n📈 Evaluation Results:")
        for model_type, accuracy in results.items():
            print(f"  {model_type.capitalize()}: {accuracy:.2%} accuracy")
        
        return results


# ========== WEB INTERFACE ==========

class WebInterface:
    """
    Simple web interface for the security suite
    Runs locally with Flask
    """
    
    def __init__(self, suite: DeepfakeSecuritySuite):
        self.suite = suite
        self.app = None
        
    def run(self, host: str = 'localhost', port: int = 5000):
        """Start the web interface"""
        from flask import Flask, render_template, request, jsonify
        
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze():
            file = request.files['file']
            file_path = f"uploads/{file.filename}"
            file.save(file_path)
            
            result = self.suite.analyze_file(file_path)
            return jsonify(self.suite.result_to_dict(result))
        
        @self.app.route('/batch', methods=['POST'])
        def batch_analyze():
            files = request.files.getlist('files')
            results = []
            
            for file in files:
                file_path = f"uploads/{file.filename}"
                file.save(file_path)
                result = self.suite.analyze_file(file_path)
                results.append(self.suite.result_to_dict(result))
            
            return jsonify(results)
        
        print(f"🌐 Starting web interface at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=True)


# ========== COMMAND LINE INTERFACE ==========

def main():
    """Main command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Deepfake Security Suite - All-in-One Detection Platform'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a file')
    analyze_parser.add_argument('file', help='File to analyze')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Analyze directory')
    batch_parser.add_argument('directory', help='Directory to analyze')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Real-time monitor')
    monitor_parser.add_argument('--source', choices=['webcam', 'microphone'],
                               default='webcam', help='Source to monitor')
    monitor_parser.add_argument('--duration', type=int, default=60,
                               help='Duration in seconds')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--port', type=int, default=5000,
                           help='Port for web interface')
    
    args = parser.parse_args()
    
    # Initialize suite
    suite = DeepfakeSecuritySuite()
    
    if args.command == 'analyze':
        result = suite.analyze_file(args.file)
        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        print(f"File Type: {result.file_type.upper()}")
        print(f"Is Deepfake: {'🚨 YES' if result.is_deepfake else '✅ NO'}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Risk Level: {result.risk_level.upper()}")
        print(f"Analysis Time: {result.analysis_duration:.2f}s")
        
        if result.indicators:
            print("\n📊 Detection Indicators:")
            for indicator in result.indicators:
                print(f"  • {indicator}")
        
        if result.recommendations:
            print("\n💡 Recommendations:")
            for rec in result.recommendations:
                print(f"  • {rec}")
    
    elif args.command == 'batch':
        results = suite.batch_analyze(args.directory)
        report = suite.generate_report([r for r in results.values() 
                                       if not isinstance(r, dict) or 'error' not in r])
        
        print("\n" + "="*50)
        print("BATCH ANALYSIS REPORT")
        print("="*50)
        print(f"Total Files: {report['summary']['total_files']}")
        print(f"Deepfakes Detected: {report['summary']['deepfakes_detected']}")
        print(f"Deepfake Rate: {report['summary']['deepfake_rate']:.2%}")
        
        # Export results
        suite.export_results([r for r in results.values() 
                            if not isinstance(r, dict) or 'error' not in r])
    
    elif args.command == 'train':
        trainer = DeepfakeModelTrainer(suite)
        trainer.generate_training_data(samples_per_type=50)
        trainer.train_all_models()
        trainer.evaluate_models()
    
    elif args.command == 'monitor':
        suite.real_time_monitor(args.source, args.duration)
    
    elif args.command == 'web':
        web_interface = WebInterface(suite)
        web_interface.run(port=args.port)
    
    else:
        # Interactive mode
        interactive_mode(suite)


def interactive_mode(suite: DeepfakeSecuritySuite):
    """Interactive command-line interface"""
    print("\n" + "="*60)
    print("🤖 DEEPFAKE SECURITY SUITE - INTERACTIVE MODE")
    print("="*60)
    
    while True:
        print("\n📋 Available Commands:")
        print("  1. Analyze single file")
        print("  2. Batch analyze directory")
        print("  3. Train models")
        print("  4. Real-time monitor")
        print("  5. View analysis history")
        print("  6. Export results")
        print("  7. Start web interface")
        print("  8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == '1':
            file_path = input("Enter file path: ").strip()
            if os.path.exists(file_path):
                result = suite.analyze_file(file_path)
                
                print(f"\n📊 Result: {'🚨 DEEPFAKE' if result.is_deepfake else '✅ Authentic'}")
                print(f"   Confidence: {result.confidence:.2%}")
                print(f"   Risk Level: {result.risk_level}")
                print(f"   Analysis Time: {result.analysis_duration:.2f}s")
            else:
                print("❌ File not found!")
        
        elif choice == '2':
            dir_path = input("Enter directory path: ").strip()
            if os.path.exists(dir_path):
                results = suite.batch_analyze(dir_path)
                print(f"\n✅ Analyzed {len(results)} files")
            else:
                print("❌ Directory not found!")
        
        elif choice == '3':
            print("\n🎯 Training all models...")
            trainer = DeepfakeModelTrainer(suite)
            trainer.generate_training_data(samples_per_type=50)
            trainer.train_all_models()
            trainer.evaluate_models()
        
        elif choice == '4':
            print("\n🎥 Real-time Monitoring")
            print("  1. Monitor webcam")
            print("  2. Monitor microphone")
            source_choice = input("Select source (1-2): ").strip()
            
            duration = input("Duration in seconds (default 60): ").strip()
            duration = int(duration) if duration.isdigit() else 60
            
            if source_choice == '1':
                suite.real_time_monitor('webcam', duration)
            elif source_choice == '2':
                suite.real_time_monitor('microphone', duration)
        
        elif choice == '5':
            if suite.analyses_history:
                print(f"\n📜 Analysis History ({len(suite.analyses_history)} entries)")
                for i, result in enumerate(suite.analyses_history[-10:], 1):
                    status = "🚨 FAKE" if result.is_deepfake else "✅ REAL"
                    print(f"  {i}. {result.file_type.upper():6} | {status:10} | "
                         f"Conf: {result.confidence:.2%} | {result.timestamp[:19]}")
            else:
                print("No analysis history yet.")
        
        elif choice == '6':
            if suite.analyses_history:
                format_choice = input("Export format (json/csv): ").strip().lower()
                if format_choice in ['json', 'csv']:
                    suite.export_results(suite.analyses_history, format_choice)
                else:
                    print("Invalid format. Use 'json' or 'csv'.")
            else:
                print("No results to export.")
        
        elif choice == '7':
            port = input("Port (default 5000): ").strip()
            port = int(port) if port.isdigit() else 5000
            
            web_interface = WebInterface(suite)
            web_interface.run(port=port)
        
        elif choice == '8':
            print("\n👋 Exiting Deepfake Security Suite. Stay secure!")
            break
        
        else:
            print("Invalid option. Please choose 1-8.")


# ========== QUICK START IMPLEMENTATION ==========

def quick_start():
    """Get started in 5 minutes with minimal setup"""
    print("🚀 DEEPFAKE SECURITY SUITE - QUICK START")
    print("="*50)
    
    # Create minimal project structure
    base_dirs = ['models', 'uploads', 'temp', 'exports']
    for dir_name in base_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Initialize suite
    suite = DeepfakeSecuritySuite()
    
    # Generate sample data if no models exist
    if not os.path.exists('models/voice_detector.pkl'):
        print("\n📊 No models found. Generating sample data...")
        
        # Create simple voice dataset
        import pyttsx3
        import soundfile as sf
        import librosa
        
        engine = pyttsx3.init()
        
        # Create real samples
        for i in range(5):
            text = f"Sample voice number {i}"
            engine.save_to_file(text, f'temp/real_{i}.wav')
            engine.runAndWait()
        
        # Create fake samples (pitch shifted)
        for i in range(5):
            y, sr = librosa.load(f'temp/real_{i}.wav')
            y_fake = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
            sf.write(f'temp/fake_{i}.wav', y_fake, sr)
        
        print("✅ Sample data created in 'temp/' directory")
    
    print("\n🎯 Ready to analyze! Run one of these commands:")
    print("  python deepfake_suite.py analyze test_audio.wav")
    print("  python deepfake_suite.py batch ./samples")
    print("  python deepfake_suite.py train")
    print("  python deepfake_suite.py monitor")
    print("\nOr run without arguments for interactive mode!")


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments - show quick start or interactive
        if not os.path.exists('models'):
            quick_start()
        else:
            # Start interactive mode
            suite = DeepfakeSecuritySuite()
            interactive_mode(suite)
    else:
        # Parse command line arguments
        main()