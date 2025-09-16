declare module 'react-cytoscapejs' {
  import { ComponentType } from 'react';
  import type { Core, CytoscapeOptions, ElementsDefinition } from 'cytoscape';
  interface Props {
    elements?: ElementsDefinition | any[];
    style?: React.CSSProperties;
    layout?: CytoscapeOptions['layout'];
    stylesheet?: any[];
    cy?: (cy: Core) => void;
  }
  const CytoscapeComponent: ComponentType<Props>;
  export default CytoscapeComponent;
}
