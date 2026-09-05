"""Quick smoke tests for entity_finder.py -- not pytest, just asserts.
Run: python scripts/lib/test_entity_finder.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.entity_finder import (
    find_class_entity, find_method_entity,
    long_method_holds, blob_holds, data_class_holds, feature_envy_holds,
)

SRC_CLASS = '''package foo;

/** javadoc with a { brace } and "string with } brace" */
public class Foo {
    private int x;
    // a comment with { unbalanced brace
    public int getX() { return x; }
    public void setX(int x) { this.x = x; }

    public class Inner {
        void innerMethod() { System.out.println("{"); }
    }
}

class AfterFoo {
}
'''

def test_find_class():
    span = find_class_entity(SRC_CLASS, "Foo")
    assert span is not None, "Foo not found"
    lines = SRC_CLASS.splitlines()
    assert lines[span.start_line - 1].strip().startswith("public class Foo"), lines[span.start_line-1]
    assert lines[span.end_line - 1].strip() == "}", lines[span.end_line-1]
    print("test_find_class OK:", span.start_line, span.end_line)

def test_find_method():
    span = find_method_entity(SRC_CLASS, "getX", 0)
    assert span is not None
    assert "return x" in span.body
    print("test_find_method OK:", span.start_line, span.end_line)

def test_find_method_overload():
    src = '''class C {
    void m() { doA(); }
    void m(int a) { doB(); }
    void m(int a, int b) { doC(); }
}'''
    span0 = find_method_entity(src, "m", 0)
    span2 = find_method_entity(src, "m", 2)
    assert "doA" in span0.body, span0.body
    assert "doC" in span2.body, span2.body
    print("test_find_method_overload OK")

def test_missing_entity():
    span = find_class_entity(SRC_CLASS, "DoesNotExist")
    assert span is None
    print("test_missing_entity OK")

def test_data_class_proxy():
    src = '''class D {
    private int x;
    private String y;
    public int getX() { return x; }
    public void setX(int x) { this.x = x; }
    public String getY() { return y; }
    public void setY(String y) { this.y = y; }
}'''
    span = find_class_entity(src, "D")
    assert data_class_holds(span) is True, "expected data class proxy to hold"
    print("test_data_class_proxy OK")

def test_long_method_proxy():
    body_lines = "\n".join(f"    int v{i} = {i};" for i in range(120))
    src = f"class E {{\n  void big() {{\n{body_lines}\n  }}\n}}"
    span = find_method_entity(src, "big", 0)
    assert span is not None
    assert long_method_holds(span) is True, (span.start_line, span.end_line)
    print("test_long_method_proxy OK:", span.end_line - span.start_line + 1, "loc")

def test_string_with_braces_does_not_break_scan():
    src = '''class F {
    String s = "unbalanced { brace in string";
    void m() { System.out.println("ok"); }
}'''
    span = find_class_entity(src, "F")
    assert span is not None
    lines = src.splitlines()
    assert lines[span.end_line - 1].strip() == "}"
    print("test_string_with_braces_does_not_break_scan OK")

if __name__ == "__main__":
    test_find_class()
    test_find_method()
    test_find_method_overload()
    test_missing_entity()
    test_data_class_proxy()
    test_long_method_proxy()
    test_string_with_braces_does_not_break_scan()
    print("\nAll smoke tests passed.")
