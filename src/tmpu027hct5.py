# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__get_p(self):
            with ag__.FunctionScope('get_p', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                try:
                    do_return = True
                    retval_ = ag__.converted_call(ag__.ld(tf).nn.sigmoid, (ag__.ld(self).p_logit[0],), None, fscope)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__get_p
    return inner_factory