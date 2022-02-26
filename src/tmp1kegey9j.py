# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__concrete_dropout(self, x):
            '\n        Concrete dropout - used at training time (gradients can be propagated)\n        :param x: input\n        :return:  approx. dropped out input\n        '
            with ag__.FunctionScope('concrete_dropout', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                x_init = ag__.converted_call(ag__.ld(tf).identity, (ag__.ld(x),), None, fscope)
                eps = ag__.converted_call(ag__.ld(K).cast_to_floatx, (ag__.converted_call(ag__.ld(K).epsilon, (), None, fscope),), None, fscope)
                temp = 0.1
                unif_noise = ag__.converted_call(ag__.ld(K).random_uniform, (), dict(shape=ag__.converted_call(ag__.ld(tf).concat, ([ag__.converted_call(ag__.ld(K).shape, (ag__.ld(x),), None, fscope)[0:1], ag__.converted_call(ag__.ld(tf).ones, ((3,),), dict(dtype=ag__.converted_call(ag__.ld(K).shape, (ag__.ld(x),), None, fscope).dtype), fscope)],), dict(axis=0), fscope)), fscope)
                drop_prob = (((ag__.converted_call(ag__.ld(K).log, ((ag__.converted_call(ag__.ld(self).get_p, (), None, fscope) + ag__.ld(eps)),), None, fscope) - ag__.converted_call(ag__.ld(K).log, (((1.0 - ag__.converted_call(ag__.ld(self).get_p, (), None, fscope)) + ag__.ld(eps)),), None, fscope)) + ag__.converted_call(ag__.ld(K).log, ((ag__.ld(unif_noise) + ag__.ld(eps)),), None, fscope)) - ag__.converted_call(ag__.ld(K).log, (((1.0 - ag__.ld(unif_noise)) + ag__.ld(eps)),), None, fscope))
                drop_prob = ag__.converted_call(ag__.ld(K).sigmoid, ((ag__.ld(drop_prob) / ag__.ld(temp)),), None, fscope)
                random_tensor = (1.0 - ag__.ld(drop_prob))
                retain_prob = (1.0 - ag__.converted_call(ag__.ld(self).get_p, (), None, fscope))
                x = ag__.ld(x)
                x *= random_tensor
                x = ag__.ld(x)
                x /= retain_prob
                try:
                    do_return = True
                    retval_ = ag__.ld(x)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__concrete_dropout
    return inner_factory