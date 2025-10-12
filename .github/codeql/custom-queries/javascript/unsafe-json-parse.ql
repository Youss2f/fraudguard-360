/**
 * @name Unsafe JSON parsing
 * @description Detects unsafe JSON.parse() usage without error handling
 * @kind problem
 * @problem.severity warning
 * @security-severity 6.0
 * @precision medium
 * @id js/unsafe-json-parse
 * @tags security
 *       maintainability
 */

import javascript

from CallExpr call, Function containingFunction
where
  call.getCalleeName() = "parse" and
  call.getReceiver().(PropAccess).getBase().getALocalName() = "JSON" and
  containingFunction = call.getEnclosingFunction() and
  not exists(TryStmt try | 
    try.getBody().getAStmt*() = call.getEnclosingStmt() or
    try.getCatchClause().getBody().getAStmt*() = call.getEnclosingStmt()
  )
select call, "JSON.parse() should be wrapped in try-catch block to handle malformed JSON"