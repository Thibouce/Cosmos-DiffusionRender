# ComfyUI imports this folder as a package; when a tool (e.g. pytest) imports the file standalone
# there is no parent package and nothing to register.
if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
