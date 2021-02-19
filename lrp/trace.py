trace_enabled=False
trace_stack=None

def do_trace(R):
    """
    if we are tracing (i.e., keeping track of hidden layer relevance, put a
    copy of relevance_input in trace_stack).
    """ 
    if trace_enabled:
        trace_stack.append(R.detach().clone())

def enable_and_clean():
    global trace_enabled,trace_stack
    trace_stack=[]
    trace_enabled=True

def collect_and_disable():
    global trace_enabled,trace_stack
    old_stack=trace_stack
    trace_stack=None
    return old_stack

