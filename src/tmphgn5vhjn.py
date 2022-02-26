# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__result(self):
            with ag__.FunctionScope('result', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                precision = ag__.converted_call(ag__.ld(tf).math.divide_no_nan, (ag__.ld(self).true_positives, (ag__.ld(self).true_positives + ag__.ld(self).false_positives)), None, fscope)
                recall = ag__.converted_call(ag__.ld(tf).math.divide_no_nan, (ag__.ld(self).true_positives, (ag__.ld(self).true_positives + ag__.ld(self).false_negatives)), None, fscope)
                mul_value = (ag__.ld(precision) * ag__.ld(recall))
                add_value = ((ag__.converted_call(ag__.ld(tf).math.square, (ag__.ld(self).beta,), None, fscope) * ag__.ld(precision)) + ag__.ld(recall))
                mean = ag__.converted_call(ag__.ld(tf).math.divide_no_nan, (ag__.ld(mul_value), ag__.ld(add_value)), None, fscope)
                f1_score = (ag__.ld(mean) * (1 + ag__.converted_call(ag__.ld(tf).math.square, (ag__.ld(self).beta,), None, fscope)))

                def get_state_1():
                    return (f1_score,)

                def set_state_1(vars_):
                    nonlocal f1_score
                    (f1_score,) = vars_

                def if_body_1():
                    nonlocal f1_score
                    weights = ag__.converted_call(ag__.ld(tf).math.divide_no_nan, (ag__.ld(self).weights_intermediate, ag__.converted_call(ag__.ld(tf).reduce_sum, (ag__.ld(self).weights_intermediate,), None, fscope)), None, fscope)
                    f1_score = ag__.converted_call(ag__.ld(tf).reduce_sum, ((ag__.ld(f1_score) * ag__.ld(weights)),), None, fscope)

                def else_body_1():
                    nonlocal f1_score

                    def get_state():
                        return (f1_score,)

                    def set_state(vars_):
                        nonlocal f1_score
                        (f1_score,) = vars_

                    def if_body():
                        nonlocal f1_score
                        f1_score = ag__.converted_call(ag__.ld(tf).reduce_mean, (ag__.ld(f1_score),), None, fscope)

                    def else_body():
                        nonlocal f1_score
                        pass
                    ag__.if_stmt((ag__.ld(self).average is not None), if_body, else_body, get_state, set_state, ('f1_score',), 1)
                weights = ag__.Undefined('weights')
                ag__.if_stmt((ag__.ld(self).average == 'weighted'), if_body_1, else_body_1, get_state_1, set_state_1, ('f1_score',), 1)
                try:
                    do_return = True
                    retval_ = ag__.ld(f1_score)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__result
    return inner_factory