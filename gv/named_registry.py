
class NamedRegistry(object):
    REGISTRY = {}

    @property
    def name(self):
        """Returns the name of the registry entry"""
        # Automatically overloaded by 'register'
        return "noname" 

    @classmethod
    def register(cls, name):
        """Decorator to register a class"""
        def register_decorator(reg_cls):
            def name_func(self):
                return name
            reg_cls.name = property(name_func)
            assert issubclass(reg_cls, cls), "Must be subclass of BinaryDescriptor"
            cls.REGISTRY[name] = reg_cls
            return reg_cls
        return register_decorator

    @classmethod
    def getclass(cls, name):
        return cls.REGISTRY[name]

    @classmethod
    def construct(cls, name, *args, **kwargs):
        return cls.REGISTRY[name](*args, **kwargs)

    @classmethod
    def registry(cls):
        return cls.REGISTRY

    @classmethod
    def root(cls, reg_cls):
        """
        Decorate your base class with this, to create
        a new registry for it
        """
        reg_cls.REGISTRY = {}
        return reg_cls 

