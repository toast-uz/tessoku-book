use std::cmp::Reverse;
use proconio::input;
use proconio::marker::Usize1;
use itertools::Itertools;
use rustc_hash::FxHashSet as HashSet;

const N: usize = 20;
const M: usize = 3;
const T: usize = 100;
const MAX_BEAM_DEPTH: usize = T;
const MAX_BEAM_WIDTH: usize = 20000;

//const LIMIT: f64 = 0.8; // 大量メモリの解放に時間がかかるので、余裕を持たせる
const DEBUG: bool = true;

#[allow(unused_macros)]
macro_rules! dbg {( $( $x:expr ),* ) => ( if DEBUG {eprintln!($( $x ),* );}) }
#[allow(unused_macros)]
macro_rules! dbg2 {( $( $x:expr ),* ) => ( if DEBUG {
    eprintln!($( $x ),* );
    println!("## {}", format!($( $x ),* ));
}) }

fn main() {
    let e = Env::new();
    let mut a = Agent::new(&e);
    a.optimize(&e);
    println!("{}", a.result(&e));
    dbg!("Counter = {}, memory = {} (valid: {})", a.counter, a.nodes.len(), a.nodes.len() - a.free_nodes.len());
    dbg!("Computed_score = {}", a.compute_score());
}

#[derive(Debug, Clone, Default)]
struct Env {
    root_state: State,
    pqr: Vec<Vec<usize>>,
}

impl Env {
    fn new() -> Self {
        input! {
            _t: usize,
            pqr: [[Usize1; M]; T],
        }
        let mut e = Self::default();
        e.init(&pqr); e
    }

    // テストが作りやすいように、newとinitを分離
    fn init(&mut self, pqr: &[Vec<usize>]) {
        // 問題入力の設定
        self.pqr = pqr.to_vec();
        self.root_state = State::new();
        // ハイパーパラメータの設定
    }
}

// 木dp
// 遷移がdp[i]からdp[i+1]へ限定される場合

#[derive(Debug, Clone, Default)]
struct Agent {
    state: State,
    node_id: usize,
    nodes: Vec<Node>,
    free_nodes: Vec<usize>, // 使われていないノードのindex
    counter: usize,
}

impl Agent {
    fn new(e: &Env) -> Self {
        let root_node = Node::new(Op::default());
        Self {
            state: e.root_state.clone(),
            node_id: 0,
            nodes: vec![root_node],
            free_nodes: Vec::new(),
            ..Self::default()
        }
    }
    // ビームサーチで探索
    fn optimize(&mut self, e: &Env) {
        for depth in 0..MAX_BEAM_DEPTH {
            let max_beam_width = if depth < MAX_BEAM_DEPTH - 1 { MAX_BEAM_WIDTH } else { 1 };
            self.move_root(e);
            for cand in self.emum_cands(e, depth)
                    .into_iter().sorted_unstable_by_key(|cand| Reverse(cand.score))
                    .take(max_beam_width) {
                self.push_child(cand);
                self.counter += 1;
            }
            self.prune(depth);
        }
        // ベストノードを登録
        self.node_id = (0..self.nodes.len())
            .filter(|&id| self.nodes[id].goal)
            .max_by_key(|&id| self.nodes[id].score).unwrap();
    }
    // dp復元
    fn restore(&self, mut node_id: usize) -> Vec<Op> {
        let mut res = Vec::new();
        while node_id != !0 {
            res.push(self.nodes[node_id].op.clone());
            node_id = self.nodes[node_id].parent;
        }
        res.pop();  // ルートノードのopは適用しない
        res.reverse();
        res
    }
    // node_idをdp復元して出力
    fn result(&self, _e: &Env) -> String {
        let ops = self.restore(self.node_id);
        ops.iter().map(|op| op.to_string()).join("\n")
    }
    // 次の操作を列挙する
    fn emum_cands(&self, e: &Env, depth: usize) -> Vec<Cand> {
        let mut res = Vec::new();
        let mut node_id = self.node_id;
        let mut state = self.state.clone();
        let mut seen = HashSet::from_iter([!0]);
        loop {
            // 深さdepthなら次のノードを候補に追加
            if self.nodes[node_id].depth == depth {
                res.extend(self.enum_cands_from_node(e, node_id, &state));
            }
            // 次のノードに移動
            if self.nodes[node_id].depth < depth && !seen.contains(&self.nodes[node_id].child) {
                // 深さ不十分で未到達な子供がいるなら、子供に移動
                let parent_depth = self.nodes[node_id].depth;
                node_id = self.nodes[node_id].child;
                seen.insert(node_id);
                state.apply(e, parent_depth, &self.nodes[node_id].op);
            } else {
                if node_id == self.node_id { break; }   // ルートノードに戻っていたら終了
                assert_ne!(self.nodes[node_id].parent, !0, "parent_id = !0, node = {} {:?} {}", node_id, self.nodes[node_id], self.free_nodes.contains(&node_id));
                let parent_depth = self.parent_depth(node_id);
                state.revert(e, parent_depth, &self.nodes[node_id].op); // ステートをもとに戻す
                if self.nodes[node_id].next != !0 {
                    // 兄弟がいるなら兄弟に移動
                    node_id = self.nodes[node_id].next;
                    state.apply(e, parent_depth, &self.nodes[node_id].op);
                } else {
                    // 兄弟がいないなら親に移動
                    node_id = self.nodes[node_id].parent;
                }
            }
        }
        res
    }
    fn enum_cands_from_node(&self, e: &Env, node_id: usize, state: &State) -> Vec<Cand> {
        let mut res = Vec::new();
        let node = &self.nodes[node_id];
        let pqr = &e.pqr[node.depth];
        let score_diff_old = (0..M).filter(|&i| state.0[pqr[i]] == 0).count() as isize;
        for op in [Op(1), Op(-1)].iter() {
            let mut cand = Cand { op: op.clone(), depth: node.depth + 1, score: node.score, score_diff: 0, parent: node_id, goal: false };
            let score_diff_new = (0..M).filter(|&i| state.0[pqr[i]] + op.0 == 0).count() as isize;
            cand.score_diff = node.score_diff + score_diff_new - score_diff_old;
            cand.score = node.score + cand.score_diff;
            cand.goal = if cand.depth == T { true } else { false };
            res.push(cand);
        }
        res
    }
    // ルートノードを分岐位置まで移動
    fn move_root(&mut self, e: &Env) {
        while self.nodes[self.node_id].child != !0 &&
                self.nodes[self.nodes[self.node_id].child].next == !0 {
            let parent_depth = self.nodes[self.node_id].depth;
            self.node_id = self.nodes[self.node_id].child;
            self.state.apply(e, parent_depth, &self.nodes[self.node_id].op);
        }
    }
    // candを子供として追加し、そのノードのidを返す
    fn push_child(&mut self, cand: Cand) -> usize {
        let parent_node_id = cand.parent;
        let new_node = cand.to_node();
        // 新しいノードのidを決めて、ノードを登録する
        let new_node_id = if let Some(free) = self.free_nodes.pop() {
            self.nodes[free] = new_node;
            free
        } else {
            self.nodes.push(new_node);
            self.nodes.len() - 1
        };
        // 親ノードの子供リストの先頭に新しいノードを追加
        if parent_node_id != !0 {
            let next_node_id = self.nodes[parent_node_id].child;
            self.nodes[parent_node_id].child = new_node_id;
            if next_node_id != !0 {
                self.nodes[next_node_id].prev = new_node_id;
                self.nodes[new_node_id].next = next_node_id;
            }
        }
        new_node_id
    }
    // 不要になったノードを削除
    // 深さdepthまでで子供がない（かつゴールしていない）ノードを再帰的に削除する
    fn prune(&mut self, depth: usize) {
        for node_id in 0..self.nodes.len() {
            if self.nodes[node_id].parent == !0 || self.nodes[node_id].depth > depth { continue; }
            self.remove_recursive(node_id);
        }
    }
    fn remove_recursive(&mut self, node_id: usize) {
        if self.nodes[node_id].child != !0 || self.nodes[node_id].goal { return; }
        // 子供がない（かつゴールしていない）ノードを再帰的に削除
        let parent_node_id = self.nodes[node_id].parent;
        let prev_node_id = self.nodes[node_id].prev;
        let next_node_id = self.nodes[node_id].next;
        if prev_node_id != !0 { self.nodes[prev_node_id].next = next_node_id; }
        if next_node_id != !0 { self.nodes[next_node_id].prev = prev_node_id; }
        if parent_node_id != !0 && self.nodes[parent_node_id].child == node_id {
            self.nodes[parent_node_id].child = next_node_id;
            self.remove_recursive(parent_node_id);
        }
        //dbg!("free_nodes_id = {} parent = {}", node_id, self.nodes[node_id].parent);
        self.nodes[node_id] = Node::new(Op::default());
        self.free_nodes.push(node_id);
    }
    fn parent_depth(&self, node_id: usize) -> usize {
        let parent_id = self.nodes[node_id].parent;
        self.nodes[parent_id].depth
    }
    fn compute_score(&self) -> isize { self.nodes[self.node_id].score }
}

#[derive(Debug, Clone, Default)]
struct State(Vec<isize>);
impl State {
    fn new() -> Self { Self { 0: vec![0; N] } }
    fn apply(&mut self, e: &Env, parent_depth: usize, op: &Op) {
        let pqr = &e.pqr[parent_depth];
        self.0[pqr[0]] += op.0;
        self.0[pqr[1]] += op.0;
        self.0[pqr[2]] += op.0;
    }
    fn revert(&mut self, e: &Env, parent_depth: usize, op: &Op) {
        let pqr = &e.pqr[parent_depth];
        self.0[pqr[0]] -= op.0;
        self.0[pqr[1]] -= op.0;
        self.0[pqr[2]] -= op.0;
    }
}

#[derive(Debug, Clone, Default)]
struct Cand {
    op: Op,  // 親からの差分オペレーション
    depth: usize,
    score_diff: isize,
    score: isize,
    parent: usize,
    goal: bool,
}
impl Cand {
    fn to_node(&self) -> Node { Node {
        op: self.op.clone(), depth: self.depth, goal: self.goal,
        score: self.score, score_diff: self.score_diff,
        parent: self.parent, child: !0, prev: !0, next: !0,
    } }
}

#[derive(Debug, Clone, Default)]
struct Node {
    op: Op,  // 親からの差分オペレーション
    depth: usize,
    score: isize,
    score_diff: isize,
    parent: usize,
    goal: bool,
    child: usize,   // 先頭の子供
    prev: usize,
    next: usize,
}

impl Node {
    fn new(op: Op) -> Self { Self {
        op, parent: !0, child: !0, prev: !0, next: !0, score_diff: N as isize, ..Self::default()
    } }
}

#[derive(Debug, Clone, Default)]
struct Op(isize);    // 1: 操作A, -1: 操作B

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", if self.0 == 1 { "A" } else { "B" })
    }
}
