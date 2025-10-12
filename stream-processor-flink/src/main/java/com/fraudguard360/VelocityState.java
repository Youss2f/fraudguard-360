package com.fraudguard360;

import java.util.ArrayList;
import java.util.List;

/**
 * Velocity state for tracking transaction patterns per user
 */
public class VelocityState {
    private List<VelocityTransaction> transactions;

    public VelocityState() {
        this.transactions = new ArrayList<>();
    }

    public List<VelocityTransaction> getTransactions() {
        return transactions;
    }

    public void setTransactions(List<VelocityTransaction> transactions) {
        this.transactions = transactions;
    }
}

/**
 * Simplified transaction for velocity tracking
 */
class VelocityTransaction {
    private long timestamp;
    private double amount;

    public VelocityTransaction(long timestamp, double amount) {
        this.timestamp = timestamp;
        this.amount = amount;
    }

    public long getTimestamp() { return timestamp; }
    public double getAmount() { return amount; }
}

/**
 * Pattern analysis utilities
 */
class PatternAnalyzer {
    
    public PatternAnalysisResult analyzePatterns(List<EnrichedTransaction> transactions) {
        PatternAnalysisResult result = new PatternAnalysisResult();
        
        // Analyze amount patterns
        analyzeAmountPatterns(transactions, result);
        
        // Analyze time patterns
        analyzeTimePatterns(transactions, result);
        
        // Analyze merchant patterns
        analyzeMerchantPatterns(transactions, result);
        
        return result;
    }
    
    private void analyzeAmountPatterns(List<EnrichedTransaction> transactions, PatternAnalysisResult result) {
        // Check for round number patterns
        long roundNumbers = transactions.stream()
            .mapToDouble(EnrichedTransaction::getAmount)
            .filter(amount -> amount % 100 == 0 && amount >= 1000)
            .count();
        
        if (roundNumbers > transactions.size() * 0.7) { // >70% round numbers
            result.addAnomaly("Suspicious round number pattern");
            result.increaseScore(0.3);
        }
        
        // Check for identical amounts
        long uniqueAmounts = transactions.stream()
            .mapToDouble(EnrichedTransaction::getAmount)
            .distinct()
            .count();
        
        if (uniqueAmounts < transactions.size() * 0.3) { // <30% unique amounts
            result.addAnomaly("Repeated identical amounts");
            result.increaseScore(0.4);
        }
    }
    
    private void analyzeTimePatterns(List<EnrichedTransaction> transactions, PatternAnalysisResult result) {
        // Check for rapid-fire transactions (< 5 seconds apart)
        int rapidTransactions = 0;
        for (int i = 1; i < transactions.size(); i++) {
            long timeDiff = Math.abs(
                transactions.get(i).getTimestamp().toEpochMilli() - 
                transactions.get(i-1).getTimestamp().toEpochMilli()
            );
            if (timeDiff < 5000) { // Less than 5 seconds
                rapidTransactions++;
            }
        }
        
        if (rapidTransactions > transactions.size() * 0.5) {
            result.addAnomaly("Rapid-fire transaction pattern");
            result.increaseScore(0.5);
        }
    }
    
    private void analyzeMerchantPatterns(List<EnrichedTransaction> transactions, PatternAnalysisResult result) {
        // Check for single merchant dominance
        long uniqueMerchants = transactions.stream()
            .map(EnrichedTransaction::getMerchantId)
            .distinct()
            .count();
        
        if (uniqueMerchants == 1 && transactions.size() > 5) {
            result.addAnomaly("All transactions to single merchant");
            result.increaseScore(0.3);
        }
    }
}

/**
 * Pattern analysis result
 */
class PatternAnalysisResult {
    private List<String> anomalies;
    private double anomalyScore;
    
    public PatternAnalysisResult() {
        this.anomalies = new ArrayList<>();
        this.anomalyScore = 0.0;
    }
    
    public void addAnomaly(String anomaly) {
        this.anomalies.add(anomaly);
    }
    
    public void increaseScore(double increase) {
        this.anomalyScore = Math.min(this.anomalyScore + increase, 1.0);
    }
    
    public boolean isAnomalous() {
        return anomalyScore > 0.3;
    }
    
    public List<String> getAnomalies() { return anomalies; }
    public double getAnomalyScore() { return anomalyScore; }
}

/**
 * ML Model Proxy for fraud scoring
 */
class MLModelProxy {
    private static final String MODEL_VERSION = "1.0.0";
    
    public double predictFraudProbability(EnrichedTransaction transaction) {
        // Simulate ML model call - in production would call actual ML service
        // For now, use weighted risk factors as proxy
        
        double score = transaction.getOverallRiskScore();
        
        // Add some ML-like complexity
        if (transaction.getAmount() > 10000 && transaction.getLocationRisk() > 0.6) {
            score += 0.2;
        }
        
        if (transaction.getDeviceRisk() > 0.7 && transaction.getTimeRisk() > 0.5) {
            score += 0.15;
        }
        
        return Math.min(score, 1.0);
    }
    
    public String getModelVersion() {
        return MODEL_VERSION;
    }
}