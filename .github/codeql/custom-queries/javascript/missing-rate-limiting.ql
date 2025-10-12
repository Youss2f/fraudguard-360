/**
 * @name Missing rate limiting middleware
 * @description Detects Express routes without rate limiting protection
 * @kind problem
 * @problem.severity warning
 * @security-severity 5.0
 * @precision medium
 * @id js/missing-rate-limiting
 * @tags security
 *       external/cwe/cwe-770
 */

import javascript

from CallExpr routeCall, string method
where
  exists(MemberAccessExpr mae | 
    mae = routeCall.getCallee() and
    mae.getBase().getALocalName() = "app" and
    method = mae.getPropertyName() and
    method.regexpMatch("get|post|put|delete|patch")
  ) and
  not exists(CallExpr middleware |
    middleware.getCalleeName().regexpMatch(".*[Rr]ate[Ll]imit.*|.*[Tt]hrottle.*") and
    middleware.getParent*() = routeCall.getParent*()
  )
select routeCall, "Route " + method.toUpperCase() + " " + routeCall.getArgument(0).toString() + " is missing rate limiting protection"