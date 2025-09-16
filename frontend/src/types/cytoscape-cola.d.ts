declare module 'cytoscape-cola' {
  import { Ext } from 'cytoscape';
  
  interface ColaOptions {
    name: 'cola';
    animate?: boolean;
    refresh?: number;
    maxSimulationTime?: number;
    ungrabifyWhileSimulating?: boolean;
    fit?: boolean;
    padding?: number;
    boundingBox?: any;
    nodeDimensionsIncludeLabels?: boolean;
    randomize?: boolean;
    avoidOverlap?: boolean;
    handleDisconnected?: boolean;
    convergenceThreshold?: number;
    nodeSpacing?: (node: any) => number;
    flow?: any;
    alignment?: any;
    gapInequalities?: any[];
    ready?: () => void;
    stop?: () => void;
  }

  const cola: Ext;
  export = cola;
}