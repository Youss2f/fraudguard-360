/**
 * @name Hardcoded credentials in environment files
 * @description Detects potential hardcoded credentials in configuration files
 * @kind problem
 * @problem.severity error
 * @security-severity 9.0
 * @precision high
 * @id js/hardcoded-credentials-env
 * @tags security
 *       external/cwe/cwe-798
 */

import javascript

from StringLiteral str, string content
where
  str.getValue() = content and
  (
    content.regexpMatch("(?i).*(password|secret|key|token|api_key)\\s*=\\s*['\"][^'\"]{8,}['\"].*") or
    content.regexpMatch("(?i).*ghp_[a-zA-Z0-9]{36}.*") or
    content.regexpMatch("(?i).*sk-[a-zA-Z0-9]{48}.*") or
    content.regexpMatch("(?i).*xox[baprs]-[a-zA-Z0-9]{10,48}.*")
  ) and
  not str.getFile().getBaseName().matches("test%") and
  not str.getFile().getBaseName().matches("spec%") and
  not str.getFile().getBaseName().matches("example%")
select str, "Potential hardcoded credential found: " + content.prefix(50)