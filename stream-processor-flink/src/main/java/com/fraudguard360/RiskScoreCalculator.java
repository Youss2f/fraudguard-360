package com.fraudguard360;

import java.time.Instant;
import java.time.LocalTime;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Risk score calculator for transaction enrichment
 */
public class RiskScoreCalculator {
    
    private static final Set<String> HIGH_RISK_CATEGORIES = new HashSet<>(Arrays.asList(
        "gambling", "adult", "cryptocurrency", "money_transfer", "prepaid_cards"
    ));
    
    private static final Set<String> MEDIUM_RISK_CATEGORIES = new HashSet<>(Arrays.asList(
        "electronics", "jewelry", "luxury_goods", "travel", "digital_goods"
    ));

    /**
     * Calculate amount-based risk score
     */
    public double calculateAmountRisk(double amount) {
        if (amount > 50000) return 0.9;
        if (amount > 20000) return 0.7;
        if (amount > 10000) return 0.5;
        if (amount > 5000) return 0.3;
        return 0.1;
    }

    /**
     * Calculate location-based risk score
     */
    public double calculateLocationRisk(Double latitude, Double longitude) {
        if (latitude == null || longitude == null) {
            return 0.6; // Missing location is medium risk
        }
        
        // Simplified location risk - in production would use geofencing service
        // Check if location is in known high-risk areas
        if (isHighRiskLocation(latitude, longitude)) {
            return 0.8;
        }
        
        return 0.2; // Default low risk for known locations
    }

    /**
     * Calculate merchant category risk
     */
    public double calculateMerchantRisk(String category) {
        if (category == null) return 0.5;
        
        String lowerCategory = category.toLowerCase();
        
        if (HIGH_RISK_CATEGORIES.contains(lowerCategory)) {
            return 0.8;
        }
        
        if (MEDIUM_RISK_CATEGORIES.contains(lowerCategory)) {
            return 0.5;
        }
        
        return 0.2; // Low risk for standard categories
    }

    /**
     * Calculate time-based risk score
     */
    public double calculateTimeRisk(Instant timestamp) {
        LocalTime time = LocalTime.ofInstant(timestamp, java.time.ZoneOffset.UTC);
        int hour = time.getHour();
        
        // Higher risk for unusual hours
        if (hour >= 2 && hour <= 6) {
            return 0.7; // Late night/early morning
        }
        
        if (hour >= 22 || hour <= 1) {
            return 0.4; // Late evening/very early morning
        }
        
        return 0.1; // Normal business hours
    }

    /**
     * Calculate device-based risk score
     */
    public double calculateDeviceRisk(String deviceFingerprint) {
        if (deviceFingerprint == null || deviceFingerprint.isEmpty()) {
            return 0.8; // Missing device info is high risk
        }
        
        // Check for suspicious device patterns
        String lower = deviceFingerprint.toLowerCase();
        
        if (lower.contains("emulator") || lower.contains("simulator")) {
            return 0.9;
        }
        
        if (lower.contains("proxy") || lower.contains("vpn")) {
            return 0.7;
        }
        
        if (lower.startsWith("unknown") || lower.contains("generic")) {
            return 0.6;
        }
        
        return 0.2; // Known device with good fingerprint
    }

    /**
     * Check if coordinates are in a high-risk geographical area
     */
    private boolean isHighRiskLocation(double latitude, double longitude) {
        // Simplified implementation - in production would use geofencing service
        // Example: certain regions or countries might be flagged as high-risk
        
        // Mock high-risk coordinates (example coordinates)
        double[][] highRiskAreas = {
            {40.7128, -74.0060}, // Example: specific city coordinates
            {51.5074, -0.1278}   // Another example location
        };
        
        for (double[] riskArea : highRiskAreas) {
            double distance = calculateDistance(latitude, longitude, riskArea[0], riskArea[1]);
            if (distance < 50) { // Within 50km of high-risk area
                return true;
            }
        }
        
        return false;
    }

    /**
     * Calculate distance between two geographical points (Haversine formula)
     */
    private double calculateDistance(double lat1, double lon1, double lat2, double lon2) {
        final int R = 6371; // Radius of the earth in km
        
        double latDistance = Math.toRadians(lat2 - lat1);
        double lonDistance = Math.toRadians(lon2 - lon1);
        
        double a = Math.sin(latDistance / 2) * Math.sin(latDistance / 2)
                + Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2))
                * Math.sin(lonDistance / 2) * Math.sin(lonDistance / 2);
        
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        
        return R * c; // Distance in km
    }
}