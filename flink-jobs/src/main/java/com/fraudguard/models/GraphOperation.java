package com.fraudguard.models;

public class GraphOperation {
    public String caller;
    public String callee;
    public int duration;

    public GraphOperation() {}

    public GraphOperation(String caller, String callee, int duration) {
        this.caller = caller;
        this.callee = callee;
        this.duration = duration;
    }
}
