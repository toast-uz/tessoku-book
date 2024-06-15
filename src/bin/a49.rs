use std::time::Instant;
use std::cmp::Reverse;
use proconio::input;
use proconio::marker::Usize1;
use itertools::Itertools;
use rustc_hash::FxHashSet as HashSet;
use xorshift_rand::*;

const N: usize = 20;
const M: usize = 3;
const T: usize = 100;
const MAX_DEPTH: usize = T;
const BEAM_WIDTH_MIN: usize = 20000;
const BEAM_WIDTH_DEFAULT: usize = 30000;
const BEAM_WIDTH_MAX: usize = 50000;

const LIMIT: f64 = 0.95;
const DEBUG: bool = true;

#[allow(unused_macros)]
macro_rules! dbg {( $( $x:expr ),* ) => ( if DEBUG {eprintln!($( $x ),* );}) }
#[allow(unused_macros)]
macro_rules! dbg2 {( $( $x:expr ),* ) => ( if DEBUG {
    eprintln!($( $x ),* );
    println!("## {}", format!($( $x ),* ));
}) }

fn main() {
    let timer = Instant::now();
    let mut rng = xorshift_rng();
    let e = Env::new(&mut rng);
    let mut a = Agent::new(&e);
    a.optimize(&e, &timer, LIMIT);
    println!("{}", a.result(&e));
    dbg!("Computed_score = {}", a.compute_score());
}

#[derive(Debug, Clone, Default)]
struct Env {
    root_state: State,
    seed: Vec<u64>,
    pqr: Vec<Vec<usize>>,
}

impl Env {
    fn new(rng: &mut XorshiftRng) -> Self {
        input! {
            _t: usize,
            pqr: [[Usize1; M]; T],
        }
        let mut e = Self::default();
        e.init(&pqr, rng); e
    }

    // テストが作りやすいように、newとinitを分離
    fn init(&mut self, pqr: &[Vec<usize>], rng: &mut XorshiftRng) {
        // 問題入力の設定
        self.pqr = pqr.to_vec();
        self.root_state = State::new();
        // 乱数シードの設定
        self.seed = (0..N).map(|_| rng.gen_u64()).collect();
        // ハイパーパラメータの設定
    }
}

// 木dp
// 遷移がdp[i]からdp[i+1]へ限定される場合

#[derive(Debug, Clone, Default)]
struct Agent {
    best_node_id: usize,
    nodes: Vec<Node>,
    free_nodes: Vec<usize>, // 使われていないノードのindex
    counter: usize,
}

impl Agent {
    fn new(e: &Env) -> Self {
        let root_node = Node::new(Op::default());
        let mut a = Self {
            nodes: vec![root_node],
            free_nodes: Vec::new(),
            ..Self::default()
        };
        a.init(e); a
    }
    fn init(&mut self, e: &Env) {
        let (cands, _) = self.emum_cands(e, 0);
        self.push_child(cands[0].clone());
    }
    // ビームサーチで探索
    fn optimize(&mut self, e: &Env, timer: &Instant, limit: f64) {
        // depth=0は、initで既に処理済み
        let mut width = BEAM_WIDTH_DEFAULT;
        for depth in 1..MAX_DEPTH {
            // 残り時間をもとにビーム幅を調整
            if depth == MAX_DEPTH - 1 { width = 1; }
            else {
                let elapsed = timer.elapsed().as_secs_f64();
                let rest_time = limit - elapsed;
                width = ((width as f64 * (depth as f64 * rest_time /
                    (MAX_DEPTH - depth) as f64 / elapsed).sqrt()) as usize)
                    .clamp(BEAM_WIDTH_MIN, BEAM_WIDTH_MAX);
            }

            let (mut cands, parent_ids) = self.emum_cands(e, depth);
            self.counter += cands.len();
            if cands.len() > width * 2 {
                cands.select_nth_unstable_by_key(width * 2, |cand| Reverse(cand.score));
            }
            cands.sort_unstable_by_key(|cand| Reverse(cand.score));
            for cand in cands.into_iter()
                    .unique_by(|cand| cand.hash)
                    .take(width) {
                self.push_child(cand);
            }
            for &node_id in &parent_ids {
                self.remove_recursive(node_id);
            }
            dbg!("#{} counter={}, width={} memory={} (valid:{} tested:{})", depth, self.counter, width, self.nodes.len(), self.nodes.len() - self.free_nodes.len(), parent_ids.len());
        }
        self.best_node_id = self.compute_best_node_id();
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
        // ベストノードからdp復元
        let ops = self.restore(self.best_node_id);
        // 出力
        ops.iter().map(|op| op.to_string()).join("\n")
    }
    fn compute_best_node_id(&self) -> usize {
        (0..self.nodes.len())
            .filter(|&id| self.nodes[id].goal)
            .max_by_key(|&id| self.nodes[id].score).unwrap()
    }
    // 次の操作を列挙する
    // 返り値: (候補, 親ノードid)
    fn emum_cands(&self, e: &Env, depth: usize) -> (Vec<Cand>, Vec<usize>) {
        let mut cands = Vec::new();
        let mut parent_ids = Vec::new();
        let mut node_id = 0;
        let mut state = e.root_state.clone();
        let mut seen = HashSet::from_iter([!0]);
        loop {
            // 深さdepthなら次のノードを候補に追加
            if self.nodes[node_id].depth == depth {
                cands.extend(self.enum_cands_from_node(e, node_id, &state));
                parent_ids.push(node_id);
            }
            // 次のノードに移動
            if self.nodes[node_id].depth < depth && !seen.contains(&self.nodes[node_id].child) {
                // 深さ不十分で未到達な子供がいるなら、子供に移動
                let parent_depth = self.nodes[node_id].depth;
                node_id = self.nodes[node_id].child;
                seen.insert(node_id);
                state.apply(e, parent_depth, &self.nodes[node_id].op);
            } else {
                if node_id == 0 { break; }   // ルートノードに戻っていたら終了
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
        (cands, parent_ids)
    }
    fn enum_cands_from_node(&self, e: &Env, node_id: usize, state: &State) -> Vec<Cand> {
        let mut res = Vec::new();
        let node = &self.nodes[node_id];
        let pqr = &e.pqr[node.depth];
        let score_diff_old = (0..M).filter(|&i| state.elms[pqr[i]] == 0).count() as isize;
        for op in [Op(1), Op(-1)].iter() {
            let mut cand = Cand {
                op: op.clone(), depth: node.depth + 1, score: node.score, parent: node_id, ..Cand::default()
            };
            let score_diff_new = (0..M).filter(|&i| state.elms[pqr[i]] + op.0 == 0).count() as isize;
            cand.score_diff = node.score_diff + score_diff_new - score_diff_old;
            cand.score = node.score + cand.score_diff;
            cand.goal = if cand.depth == T { true } else { false };
            cand.hash = state.hash.wrapping_add(state.compute_hash_diff(e, node.depth, op));
            res.push(cand);
        }
        res
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
    // 親があって子供がない（かつゴールしていない）ノードを再帰的に削除する
    fn remove_recursive(&mut self, node_id: usize) {
        if self.nodes[node_id].parent == !0 || self.nodes[node_id].child != !0 || self.nodes[node_id].goal { return; }
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
    fn compute_score(&self) -> isize { self.nodes[self.best_node_id].score }
}

#[derive(Debug, Clone, Default)]
struct State{
    elms: Vec<i8>,
    hash: u64,
}
impl State {
    fn new() -> Self { Self { elms: vec![0; N], ..Default::default() } }
    fn compute_hash_diff(&self, e: &Env, depth: usize, op: &Op) -> u64 {
        (e.seed[e.pqr[depth][0]] as i64)
            .wrapping_add(e.seed[e.pqr[depth][1]] as i64)
            .wrapping_add(e.seed[e.pqr[depth][2]] as i64)
            .wrapping_mul(op.0 as i64)
        as u64
    }
    fn apply(&mut self, e: &Env, parent_depth: usize, op: &Op) {
        let pqr = &e.pqr[parent_depth];
        self.elms[pqr[0]] += op.0;
        self.elms[pqr[1]] += op.0;
        self.elms[pqr[2]] += op.0;
        self.hash = self.hash.wrapping_add(self.compute_hash_diff(e, parent_depth, op));
    }
    fn revert(&mut self, e: &Env, parent_depth: usize, op: &Op) {
        let pqr = &e.pqr[parent_depth];
        self.elms[pqr[0]] -= op.0;
        self.elms[pqr[1]] -= op.0;
        self.elms[pqr[2]] -= op.0;
        self.hash = self.hash.wrapping_sub(self.compute_hash_diff(e, parent_depth, op));
    }
}

#[derive(Debug, Clone, Default)]
struct Cand {
    op: Op,  // 親からの差分オペレーション
    depth: usize,
    score_diff: isize,
    score: isize,
    parent: usize,
    hash: u64,
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
struct Op(i8);    // 1: 操作A, -1: 操作B

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", if self.0 == 1 { "A" } else { "B" })
    }
}


mod xorshift_rand {
    #![allow(dead_code)]
    use std::time::SystemTime;
    use rustc_hash::FxHashSet as HashSet;

    pub fn xorshift_rng() -> XorshiftRng {
        let seed = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
            .unwrap().as_secs() as u64;
        let mut rng = XorshiftRng::from_seed(seed);
        for _ in 0..100 { rng._xorshift(); }    // 初期値が偏らないようにウォーミングアップ
        rng
    }
    pub struct XorshiftRng { seed: u64, }

    impl XorshiftRng {
        pub fn from_seed(seed: u64) -> Self { Self { seed, } }
        fn _xorshift(&mut self) {
            self.seed ^= self.seed << 3;
            self.seed ^= self.seed >> 35;
            self.seed ^= self.seed << 14;
        }
        // [low, high) の範囲のusizeの乱数を求める
        pub fn gen_range<R: std::ops::RangeBounds<usize>>(&mut self, range: R) -> usize {
            let (start, end) = Self::unsafe_decode_range_(&range);
            assert!(start < end);
            self._xorshift();
            (start as u64 + self.seed % (end - start) as u64) as usize
        }
        // 重み付きで乱数を求める
        pub fn gen_range_weighted<R: std::ops::RangeBounds<usize>>(&mut self, range: R, weights: &[usize]) -> usize {
            let (start, end) = Self::unsafe_decode_range_(&range);
            assert_eq!(end - start, weights.len());
            let sum = weights.iter().sum::<usize>();
            let x = self.gen_range(0..sum);
            let mut acc = 0;
            for i in 0..weights.len() {
                acc += weights[i];
                if acc > x { return i; }
            }
            unreachable!()
        }
        // [low, high) の範囲から重複なくm個のusizeの乱数を求める
        pub fn gen_range_multiple<R: std::ops::RangeBounds<usize>>(&mut self, range: R, m: usize) -> Vec<usize> {
            let (start, end) = Self::unsafe_decode_range_(&range);
            assert!(m <= end - start);
            let many = m > (end - start) / 2; // mが半分より大きいか
            let n = if many { end - start - m } else { m };
            let mut res = HashSet::default();
            while res.len() < n {   // 半分より小さい方の数をランダムに選ぶ
                self._xorshift();
                let x = (start as u64 + self.seed % (end - start) as u64) as usize;
                res.insert(x);
            }
            (start..end).filter(|&x| many ^ res.contains(&x)).collect()
        }
        // rangeをもとに半開区間の範囲[start, end)を求める
        fn unsafe_decode_range_<R: std::ops::RangeBounds<usize>>(range: &R) -> (usize, usize) {
            let std::ops::Bound::Included(&start) = range.start_bound() else { panic!(); };
            let end = match range.end_bound() {
                std::ops::Bound::Included(&x) => x + 1,
                std::ops::Bound::Excluded(&x) => x,
                _ => panic!(),
            };
            (start, end)
        }
        // [0, 1] の範囲のf64の乱数を求める
        pub fn gen(&mut self) -> f64 {
            self._xorshift();
            self.seed as f64 / u64::MAX as f64
        }
        // u64の乱数を求める
        pub fn gen_u64(&mut self) -> u64 {
            self._xorshift();
            self.seed
        }
    }

    pub trait SliceXorshiftRandom<T> {
        fn choose(&self, rng: &mut XorshiftRng) -> T;
        fn choose_multiple(&self, rng: &mut XorshiftRng, m: usize) -> Vec<T>;
        fn choose_weighted(&self, rng: &mut XorshiftRng, weights: &[usize]) -> T;
        fn shuffle(&mut self, rng: &mut XorshiftRng);
    }

    impl<T: Clone> SliceXorshiftRandom<T> for [T] {
        fn choose(&self, rng: &mut XorshiftRng) -> T {
            let x = rng.gen_range(0..self.len());
            self[x].clone()
        }
        fn choose_weighted(&self, rng: &mut XorshiftRng, weights: &[usize]) -> T {
            let x = rng.gen_range_weighted(0..self.len(), weights);
            self[x].clone()
        }
        fn choose_multiple(&self, rng: &mut XorshiftRng, m: usize) -> Vec<T> {
            let selected = rng.gen_range_multiple(0..self.len(), m);
            selected.iter().map(|&i| self[i].clone()).collect()
        }
        fn shuffle(&mut self, rng: &mut XorshiftRng) {
            // Fisher-Yates shuffle
            for i in (1..self.len()).rev() {
                let x = rng.gen_range(0..=i);
                self.swap(i, x);
            }
        }
    }
}
