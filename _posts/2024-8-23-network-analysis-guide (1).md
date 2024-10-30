---
layout: post
title:  "Network Traffic Analysis with Python: A Practical Guide
"
date:   2024-10-28 
categories: AI 
---
# Network Traffic Analysis with Python: A Practical Guide

## Introduction
Network traffic analysis is crucial for:
- Detecting security threats
- Optimizing network performance
- Understanding user behavior

In this guide, we'll explore practical approaches to analyze network traffic using Python.

## 1. Basic Packet Capture and Analysis

First, let's capture and analyze network packets:

```python
from scapy.all import *
from collections import Counter
import pandas as pd
import numpy as np
from datetime import datetime

class NetworkAnalyzer:
    def __init__(self):
        self.packets = []
        self.flow_data = {}
    
    def capture_packets(self, duration=60):
        print(f"Capturing packets for {duration} seconds...")
        packets = sniff(timeout=duration)
        self.packets = packets
        return len(packets)
    
    def analyze_basic_stats(self):
        protocols = Counter()
        ip_sources = Counter()
        ip_destinations = Counter()
        
        for packet in self.packets:
            if IP in packet:
                protocols[packet[IP].proto] += 1
                ip_sources[packet[IP].src] += 1
                ip_destinations[packet[IP].dst] += 1
        
        return {
            'protocol_stats': protocols,
            'source_ips': ip_sources,
            'dest_ips': ip_destinations
        }
    
    def extract_flow_features(self):
        flows = {}
        
        for packet in self.packets:
            if IP in packet and (TCP in packet or UDP in packet):
                if TCP in packet:
                    sport, dport = packet[TCP].sport, packet[TCP].dport
                    flags = packet[TCP].flags
                else:
                    sport, dport = packet[UDP].sport, packet[UDP].dport
                    flags = 0
                
                flow_tuple = (packet[IP].src, packet[IP].dst, sport, dport, packet[IP].proto)
                
                if flow_tuple not in flows:
                    flows[flow_tuple] = {
                        'bytes': 0,
                        'packets': 0,
                        'start_time': packet.time,
                        'end_time': packet.time,
                        'flags': set()
                    }
                
                flows[flow_tuple]['bytes'] += len(packet)
                flows[flow_tuple]['packets'] += 1
                flows[flow_tuple]['end_time'] = packet.time
                if TCP in packet:
                    flows[flow_tuple]['flags'].add(flags)
        
        # Convert flows to feature vectors
        flow_features = []
        for flow_tuple, flow_data in flows.items():
            duration = flow_data['end_time'] - flow_data['start_time']
            feature_vector = {
                'src_ip': flow_tuple[0],
                'dst_ip': flow_tuple[1],
                'src_port': flow_tuple[2],
                'dst_port': flow_tuple[3],
                'protocol': flow_tuple[4],
                'duration': duration,
                'bytes': flow_data['bytes'],
                'packets': flow_data['packets'],
                'bytes_per_second': flow_data['bytes'] / duration if duration > 0 else 0,
                'packets_per_second': flow_data['packets'] / duration if duration > 0 else 0,
                'avg_packet_size': flow_data['bytes'] / flow_data['packets']
            }
            flow_features.append(feature_vector)
        
        self.flow_data = pd.DataFrame(flow_features)
        return self.flow_data

# Example usage
analyzer = NetworkAnalyzer()
num_packets = analyzer.capture_packets(duration=30)
print(f"Captured {num_packets} packets")

basic_stats = analyzer.analyze_basic_stats()
print("\nProtocol Statistics:")
for proto, count in basic_stats['protocol_stats'].most_common():
    print(f"Protocol {proto}: {count} packets")

flow_features = analyzer.extract_flow_features()
print("\nFlow Features:")
print(flow_features.describe())
```

## 2. Machine Learning for Traffic Classification

Now let's use ML to classify network traffic:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

class TrafficClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100)
        self.feature_columns = ['duration', 'bytes', 'packets', 
                               'bytes_per_second', 'packets_per_second', 
                               'avg_packet_size']
    
    def prepare_data(self, flow_data):
        # Assuming we have some labeled data
        # In reality, you'd need to label your flows (e.g., normal, attack, etc.)
        flow_data['label'] = flow_data.apply(self._label_flow, axis=1)
        
        X = flow_data[self.feature_columns]
        y = flow_data['label']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def _label_flow(self, flow):
        # This is a simplified labeling function
        # In reality, you'd need more sophisticated rules or manual labeling
        if flow['bytes_per_second'] > 1000000:  # 1 MB/s
            return 'high_traffic'
        elif flow['dst_port'] in [80, 443]:
            return 'web_traffic'
        else:
            return 'other'
    
    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        return classification_report(y_test, y_pred)
    
    def save_model(self, filename):
        joblib.dump((self.scaler, self.model), filename)
    
    @classmethod
    def load_model(cls, filename):
        classifier = cls()
        classifier.scaler, classifier.model = joblib.load(filename)
        return classifier

# Example usage
analyzer = NetworkAnalyzer()
analyzer.capture_packets(duration=60)
flow_data = analyzer.extract_flow_features()

classifier = TrafficClassifier()
X_train, X_test, y_train, y_test = classifier.prepare_data(flow_data)

classifier.train(X_train, y_train)
evaluation_report = classifier.evaluate(X_test, y_test)
print("\nTraffic Classification Report:")
print(evaluation_report)

# Save the model for future use
classifier.save_model('traffic_classifier.joblib')
```

## 3. Anomaly Detection in Network Traffic

Let's implement anomaly detection to find unusual network behavior:

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

class NetworkAnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.feature_columns = ['bytes_per_second', 'packets_per_second', 
                               'avg_packet_size']
    
    def train(self, flow_data):
        X = flow_data[self.feature_columns]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
    
    def detect_anomalies(self, flow_data):
        X = flow_data[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # -1 indicates anomaly, 1 indicates normal
        flow_data['is_anomaly'] = predictions == -1
        return flow_data[flow_data['is_anomaly']]
    
    def calculate_anomaly_scores(self, flow_data):
        X = flow_data[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)
        return scores

class RealTimeTrafficMonitor:
    def __init__(self, analyzer, classifier, anomaly_detector):
        self.analyzer = analyzer
        self.classifier = classifier
        self.anomaly_detector = anomaly_detector
        self.baseline_stats = None
    
    def establish_baseline(self, duration=300):
        print(f"Establishing baseline over {duration} seconds...")
        self.analyzer.capture_packets(duration=duration)
        flow_data = self.analyzer.extract_flow_features()
        
        self.baseline_stats = {
            'avg_bytes_per_second': flow_data['bytes_per_second'].mean(),
            'avg_packets_per_second': flow_data['packets_per_second'].mean(),
            'std_bytes_per_second': flow_data['bytes_per_second'].std(),
            'std_packets_per_second': flow_data['packets_per_second'].std()
        }
        
        self.anomaly_detector.train(flow_data)
        return self.baseline_stats
    
    def monitor_traffic(self, duration=60):
        print(f"Monitoring traffic for {duration} seconds...")
        self.analyzer.capture_packets(duration=duration)
        flow_data = self.analyzer.extract_flow_features()
        
        # Classify traffic
        classifications = self.classifier.predict(flow_data[self.classifier.feature_columns])
        flow_data['classification'] = classifications
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(flow_data)
        anomaly_scores = self.anomaly_detector.calculate_anomaly_scores(flow_data)
        flow_data['anomaly_score'] = anomaly_scores
        
        return {
            'flow_data': flow_data,
            'anomalies': anomalies,
            'summary': {
                'total_flows': len(flow_data),
                'anomaly_flows': len(anomalies),
                'traffic_types': Counter(classifications)
            }
        }

# Example usage
analyzer = NetworkAnalyzer()
classifier = TrafficClassifier.load_model('traffic_classifier.joblib')
anomaly_detector = NetworkAnomalyDetector()

monitor = RealTimeTrafficMonitor(analyzer, classifier, anomaly_detector)
baseline = monitor.establish_baseline(duration=120)
print("\nBaseline Statistics:")
for key, value in baseline.items():
    print(f"{key}: {value}")

monitoring_result = monitor.monitor_traffic(duration=60)
print("\nMonitoring Results:")
print(f"Total Flows: {monitoring_result['summary']['total_flows']}")
print(f"Anomalous Flows: {monitoring_result['summary']['anomaly_flows']}")
print("\nTraffic Types:")
for traffic_type, count in monitoring_result['summary']['traffic_types'].items():
    print(f"{traffic_type}: {count} flows")

if not monitoring_result['anomalies'].empty:
    print("\nTop Anomalies:")
    print(monitoring_result['anomalies'].sort_values('anomaly_score', ascending=True).head())
```

## 4. Visualization of Network Traffic

Let's create visualizations to help understand the traffic patterns:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class TrafficVisualizer:
    def __init__(self):
        plt.style.use('seaborn')
    
    def plot_traffic_volume_over_time(self, flow_data):
        plt.figure(figsize=(12, 6))
        flow_data['timestamp'] = pd.to_datetime(flow_data['start_time'], unit='s')
        hourly_traffic = flow_data.set_index('timestamp').resample('1min').sum()
        
        plt.plot(hourly_traffic.index, hourly_traffic['bytes'])
        plt.title('Network Traffic Volume Over Time')
        plt.xlabel('Time')
        plt.ylabel('Bytes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt
    
    def plot_protocol_distribution(self, flow_data):
        plt.figure(figsize=(10, 6))
        protocol_counts = flow_data['protocol'].value_counts()
        protocol_counts.plot(kind='bar')
        plt.title('Protocol Distribution')
        plt.xlabel('Protocol')
        plt.ylabel('Count')
        plt.tight_layout()
        return plt
    
    def plot_anomaly_visualization(self, flow_data):
        plt.figure(figsize=(10, 6))
        plt.scatter(flow_data['bytes_per_second'], 
                   flow_data['packets_per_second'], 
                   c=flow_data['anomaly_score'], 
                   cmap='viridis')
        plt.colorbar(label='Anomaly Score')
        plt.xlabel('Bytes per Second')
        plt.ylabel('Packets per Second')
        plt.title('Network Flow Anomalies')
        plt.tight_layout()
        return plt
    
    def plot_3d_traffic_visualization(self, flow_data):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(flow_data['bytes_per_second'],
                            flow_data['packets_per_second'],
                            flow_data['avg_packet_size'],
                            c=flow_data['anomaly_score'],
                            cmap='viridis')
        
        ax.set_xlabel('Bytes per Second')
        ax.set_ylabel('Packets per Second')
        ax.set_zlabel('Avg Packet Size')
        plt.colorbar(scatter, label='Anomaly Score')
        plt.title('3D Network Traffic Visualization')
        return plt

# Example usage
visualizer = TrafficVisualizer()

# Get monitoring results
monitoring_result = monitor.monitor_traffic(duration=60)
flow_data = monitoring_result['flow_data']

# Create visualizations
volume_plot = visualizer.plot_traffic_volume_over_time(flow_data)
volume_plot.savefig('traffic_volume.png')

protocol_plot = visualizer.plot_protocol_distribution(flow_data)
protocol_plot.savefig('protocol_distribution.png')

anomaly_plot = visualizer.plot_anomaly_visualization(flow_data)
anomaly_plot.savefig('anomaly_visualization.png')

plot_3d = visualizer.plot_3d_traffic_visualization(flow_data)
plot_3d.savefig('3d_traffic_visualization.png')
```

## 5. Practical Considerations and Best Practices

1. **Performance Optimization**
```python
# Use PyShark for better performance with large packet captures
import pyshark

def capture_with_pyshark(interface, duration):
    capture = pyshark.LiveCapture(interface=interface)
    capture.sniff(timeout=duration)
    return capture

# Use multiprocessing for faster analysis
from multiprocessing import Pool

def analyze_packet_chunk(packets):
    # Analysis code here
    pass

def parallel_analysis(all_packets, num_processes=4):
    chunk_size =
