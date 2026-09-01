def get_python_statement_classification(statement):
    if is_try_statement(statement):
        return 'try'
    elif is_break_statement(statement):
        return 'break'
    elif is_finally_statement(statement):
        return 'finally'
    elif is_continue_statement(statement):
        return 'continue'
    elif is_return_statement(statement):
        return 'return'
    elif is_annotation(statement):
        return 'annotation'
    elif is_while_statement(statement):
        return 'while'
    elif is_for_statement(statement):
        return 'for'
    elif is_if_statement(statement):
        return 'if'
    elif is_expression(statement):
        return 'expression'
    elif is_method(statement):
        return 'method'
    elif is_variable(statement):
        return 'variable'
    elif is_function_caller(statement):
        return 'function'
    elif is_import_statement(statement):
        return 'import'
    return 'None'

def is_try_statement(statement):
    if statement.startswith('try') :
        return True
    return False


def is_break_statement(statement):
    if statement.startswith('break'):
        return True
    return False


def is_finally_statement(statement):
    if statement.startswith('finally'):
        return True
    return False


def is_continue_statement(statement):
    if statement.startswith('continue'):
        return True
    return False


def is_return_statement(statement):
    if statement.startswith('return'):
        return True
    return False


def is_annotation(statement):
    if statement.startswith('#'):
        return True
    return False


def is_while_statement(statement):
    if statement.startswith('while'):
        return True
    return False


def is_for_statement(statement):
    if statement.startswith('for'):
        return True
    return False


def is_if_statement(statement):
    if statement.startswith('if') or statement.startswith('elif'):
        return True
    return False


def is_expression(statement):
    if '+' in statement or '-' in statement or '*' in statement or '/' in statement or '%' in statement or '+=' in statement or '-=' in statement \
            or '*=' in statement or '/=' in statement:
        return True
    return False


def is_method(statement):
    if statement.startswith('def'):
        return True
    return False


def is_variable(statement):
    if '=' in statement:
        return True
    return False


def is_function_caller(statement):
    if '(' in statement and statement[-1] == ')':
        return True
    return False

def is_import_statement(statement):
    if 'import' in statement:
        return True
    return False
