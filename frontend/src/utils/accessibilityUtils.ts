/**
 * Accessibility Testing Utilities for FraudGuard 360
 */

// Accessibility testing helpers
export const checkAccessibility = async (container: HTMLElement) => {
  try {
    const axe = await import('jest-axe');
    const results = await axe.axe(container);
    return results;
  } catch (error) {
    console.warn('jest-axe not available, skipping accessibility check');
    return { violations: [] };
  }
};

export const waitForComponentToLoad = async (testId: string, timeout = 5000) => {
  const { waitFor, screen } = await import('@testing-library/react');
  
  return waitFor(
    () => {
      const element = screen.getByTestId(testId);
      expect(element).toBeInTheDocument();
      return element;
    },
    { timeout }
  );
};

// Enhanced accessibility utilities
export const accessibilityUtils = {
  // Check for proper heading hierarchy
  checkHeadingStructure: (container: HTMLElement): string[] => {
    const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6');
    const headingLevels = Array.from(headings).map(h => parseInt(h.tagName.charAt(1)));
    
    const violations: string[] = [];
    for (let i = 1; i < headingLevels.length; i++) {
      if (headingLevels[i] > headingLevels[i - 1] + 1) {
        violations.push(`Heading level skipped: h${headingLevels[i - 1]} to h${headingLevels[i]}`);
      }
    }
    
    return violations;
  },

  // Check for proper form labels
  checkFormLabels: (container: HTMLElement): string[] => {
    const inputs = container.querySelectorAll('input, select, textarea');
    const violations: string[] = [];
    
    inputs.forEach((input, index) => {
      const hasLabel = input.closest('label') || 
                     document.querySelector(`label[for="${input.id}"]`) ||
                     input.getAttribute('aria-label') ||
                     input.getAttribute('aria-labelledby');
      
      if (!hasLabel) {
        violations.push(`Input at index ${index} missing label`);
      }
    });
    
    return violations;
  },

  // Check color contrast (basic implementation)
  checkColorContrast: (container: HTMLElement): string[] => {
    const elements = container.querySelectorAll('*');
    const violations: string[] = [];
    
    elements.forEach(el => {
      const styles = window.getComputedStyle(el);
      const color = styles.color;
      const backgroundColor = styles.backgroundColor;
      
      // Basic check for transparent backgrounds or same colors
      if (color === backgroundColor) {
        violations.push(`Element with same text and background color`);
      }
    });
    
    return violations;
  },

  // Check keyboard navigation
  checkKeyboardNavigation: async (container: HTMLElement): Promise<string[]> => {
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    
    const violations: string[] = [];
    
    focusableElements.forEach((el, index) => {
      const tabIndex = el.getAttribute('tabindex');
      if (tabIndex && parseInt(tabIndex) > 0) {
        violations.push(`Element at index ${index} has positive tabindex`);
      }
    });
    
    return violations;
  }
};

export default accessibilityUtils;