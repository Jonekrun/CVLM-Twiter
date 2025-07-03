
import sys as _sys
# 为向后兼容确保 `import llava` 能够正确引用此包
_sys.modules.setdefault('llava', _sys.modules[__name__])
