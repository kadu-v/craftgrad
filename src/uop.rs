use std::ops::Deref;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ops {
    Const,
}

#[derive(Clone, Copy, Debug)]
pub struct ConstFloat(f64);

impl ConstFloat {
    pub fn new(value: f64) -> Self {
        Self(if value.is_nan() {
            f64::from_bits(0x7ff8_0000_0000_0000)
        } else {
            value
        })
    }

    pub const fn value(self) -> f64 {
        self.0
    }
}

impl PartialEq for ConstFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 || (self.0.is_nan() && other.0.is_nan())
    }
}

impl Eq for ConstFloat {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arg {
    Float(ConstFloat),
}

#[derive(Debug)]
pub struct Node {
    pub op: Ops,
    pub src: Vec<UOp>,
    pub arg: Arg,
}

#[derive(Clone, Debug)]
pub struct UOp(Rc<Node>);

impl Deref for UOp {
    type Target = Node;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl UOp {
    pub fn new(op: Ops, src: Vec<UOp>, arg: Arg) -> Self {
        Self(Rc::new(Node { op, src, arg }))
    }

    pub fn ptr_eq(&self, other: &UOp) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    fn visit(node: &UOp, order: &mut Vec<UOp>) {
        for child in &node.src {
            Self::visit(child, order)
        }
        order.push(node.clone())
    }

    pub fn postorder(&self) -> Vec<Self> {
        let mut order = vec![];
        Self::visit(self, &mut order);
        order
    }

    pub fn rewrite(&self, rule: impl Fn(&Self) -> Option<Self>) -> Self {
        let mut rewritten: Vec<(Self, Self)> = vec![];
        for node in self.postorder() {
            let src = node
                .src
                .iter()
                .map(|child| {
                    for (old, _) in &rewritten {
                        if old.ptr_eq(child) {
                            return old.clone();
                        }
                    }
                    unreachable!("[Internal compile error] Must find the old node of ")
                })
                .collect::<Vec<_>>();
            let unchanged = src.iter().zip(&node.src).all(|(a, b)| a.ptr_eq(b));
            let rebuilt = if unchanged {
                node.clone()
            } else {
                Self::new(node.op, src, node.arg.clone())
            };
            let result = rule(&rebuilt).unwrap_or(rebuilt);
            rewritten.push((node, result))
        }
        rewritten
            .pop()
            .expect("[Internal compile error] Rewritten Node must have one element at least")
            .1
    }

    pub fn f32(value: f32) -> Self {
        Self::new(Ops::Const, vec![], Arg::Float(ConstFloat(value.into())))
    }
}
