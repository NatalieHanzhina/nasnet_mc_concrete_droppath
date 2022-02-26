# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__update_state(self, y_true, y_pred, sample_weight=None):
            with ag__.FunctionScope('update_state', 'fscope', ag__.STD) as fscope:

                def get_state():
                    return (y_pred,)

                def set_state(vars_):
                    nonlocal y_pred
                    (y_pred,) = vars_

                def if_body():
                    nonlocal y_pred
                    threshold = ag__.converted_call(ag__.ld(tf).reduce_max, (ag__.ld(y_pred),), dict(axis=(- 1), keepdims=True), fscope)
                    y_pred = ag__.converted_call(ag__.ld(tf).logical_and, ((ag__.ld(y_pred) >= ag__.ld(threshold)), (ag__.converted_call(ag__.ld(tf).abs, (ag__.ld(y_pred),), None, fscope) > 1e-12)), None, fscope)

                def else_body():
                    nonlocal y_pred
                    y_pred = (ag__.ld(y_pred) > ag__.ld(self).threshold)
                threshold = ag__.Undefined('threshold')
                ag__.if_stmt((ag__.ld(self).threshold is None), if_body, else_body, get_state, set_state, ('y_pred',), 1)
                y_true = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(y_true), ag__.ld(tf).int32), None, fscope)
                y_pred = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(y_pred), ag__.ld(tf).int32), None, fscope)

                @ag__.autograph_artifact
                def _count_non_zero(val):
                    with ag__.FunctionScope('_count_non_zero', 'fscope_1', ag__.STD) as fscope_1:
                        do_return_1 = False
                        retval__1 = ag__.UndefinedReturnValue()
                        non_zeros = ag__.converted_call(ag__.ld(tf).math.count_nonzero, (ag__.ld(val),), dict(axis=ag__.ld(self).axis), fscope_1)
                        try:
                            do_return_1 = True
                            retval__1 = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(non_zeros), ag__.ld(self).dtype), None, fscope_1)
                        except:
                            do_return_1 = False
                            raise
                        return fscope_1.ret(retval__1, do_return_1)
                ag__.converted_call(ag__.ld(self).true_positives.assign_add, (ag__.converted_call(ag__.ld(_count_non_zero), ((ag__.ld(y_pred) * ag__.ld(y_true)),), None, fscope),), None, fscope)
                ag__.converted_call(ag__.ld(self).false_positives.assign_add, (ag__.converted_call(ag__.ld(_count_non_zero), ((ag__.ld(y_pred) * (ag__.ld(y_true) - 1)),), None, fscope),), None, fscope)
                ag__.converted_call(ag__.ld(self).false_negatives.assign_add, (ag__.converted_call(ag__.ld(_count_non_zero), (((ag__.ld(y_pred) - 1) * ag__.ld(y_true)),), None, fscope),), None, fscope)
                ag__.converted_call(ag__.ld(self).weights_intermediate.assign_add, (ag__.converted_call(ag__.ld(_count_non_zero), (ag__.ld(y_true),), None, fscope),), None, fscope)
        return tf__update_state
    return inner_factory