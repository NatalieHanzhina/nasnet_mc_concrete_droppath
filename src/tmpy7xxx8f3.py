# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__call(self, inputs, training=None):
            with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()

                def get_state():
                    return (do_return, retval_)

                def set_state(vars_):
                    nonlocal do_return, retval_
                    (do_return, retval_) = vars_

                def if_body():
                    nonlocal do_return, retval_
                    try:
                        do_return = True
                        retval_ = ag__.converted_call(ag__.ld(self).concrete_dropout, (ag__.ld(inputs),), None, fscope)
                    except:
                        do_return = False
                        raise

                def else_body():
                    nonlocal do_return, retval_

                    @ag__.autograph_artifact
                    def relaxed_dropped_inputs():
                        with ag__.FunctionScope('relaxed_dropped_inputs', 'fscope_1', ag__.STD) as fscope_1:
                            do_return_1 = False
                            retval__1 = ag__.UndefinedReturnValue()
                            try:
                                do_return_1 = True
                                retval__1 = ag__.converted_call(ag__.ld(self).concrete_dropout, (ag__.ld(inputs),), None, fscope_1)
                            except:
                                do_return_1 = False
                                raise
                            return fscope_1.ret(retval__1, do_return_1)
                    try:
                        do_return = True
                        retval_ = ag__.converted_call(ag__.ld(K).in_train_phase, (ag__.ld(relaxed_dropped_inputs), ag__.ld(inputs)), dict(training=ag__.ld(training)), fscope)
                    except:
                        do_return = False
                        raise
                relaxed_dropped_inputs = ag__.Undefined('relaxed_dropped_inputs')
                ag__.if_stmt(ag__.ld(self).is_mc_dropout, if_body, else_body, get_state, set_state, ('do_return', 'retval_'), 2)
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory