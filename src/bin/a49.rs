use std::time::Instant;
use proconio::input;
use proconio::marker::Usize1;
use itertools::Itertools;
use xorshift_rand::*;
//use rustc_hash::FxHashMap as HashMap;

const N: usize = 20;
const M: usize = 3;
const MAX_BEAM_WIDTH: usize = 10000;

const LIMIT: f64 = 0.0;
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
    let mut rng = xorshift_rand::xorshift_rng();
    let e = Env::new();
    let mut a = Agent::new(&e);
    a.beam_search(&e, &mut rng, &timer, LIMIT);
    println!("{}", a.result());
    dbg!("counter = {}", a.counter);
    dbg!("Computed_score = {}", a.score);
}

#[derive(Debug, Clone, Default)]
struct Env {
    t: usize,
    pqr: Vec<Vec<usize>>,
}

impl Env {
    fn new() -> Self {
        input! {
            t: usize,
            pqr: [[Usize1; M]; t],
        }
        let mut e = Self::default();
        e.init(t, &pqr); e
    }

    // テストが作りやすいように、newとinitを分離
    fn init(&mut self, t: usize, pqr: &[Vec<usize>]) {
        // 問題入力の設定
        self.t = t;
        self.pqr = pqr.to_vec();
        // ハイパーパラメータの設定
    }
}

#[derive(Debug, Clone, Default)]
struct Agent {
    ans: Vec<bool>,
    score: isize,
    counter: usize,
}

impl Agent {
    fn new(_e: &Env) -> Self { Self::default() }

    fn beam_search(&mut self, e: &Env, _rng: &mut XorshiftRng, _timer: &Instant, _limit: f64) {
        // 初期状態を登録
        let mut todo = vec![State::default()];
        // ビームサーチ
        for t in 0..e.t {
            let mut next_todo = Vec::new();
            // 枝刈りのための準備
            let mut max_score = 0;
            // 現在の状態を順番に取り出す
            while let Some(state) = todo.pop() {
                // 次の状態を列挙
                for next_state in state.neighbors(e, t) {
                    // 枝刈り
                    if max_score < next_state.score {
                        dbg!("#{}, new max score from:{} to:{}", t, state.score, next_state.score);
                        max_score = next_state.score;
                    } else if max_score - next_state.score >= M as isize {
                        continue;
                    }
                    // 処理候補に登録
                    self.counter += 1;
                    next_todo.push(next_state);
                }
            }
            // 処理候補をスコアが大きい順にソートして、ビーム幅分だけ残す
            next_todo.sort_by_key(|state| -state.score);
            dbg!("#{}, next_todo size:{} score 1st:{} beam_last:{} last:{}",
                t, next_todo.len(), next_todo[0].score,
                next_todo[MAX_BEAM_WIDTH.min(next_todo.len()) - 1].score,
                next_todo[next_todo.len() - 1].score);
            next_todo.truncate(MAX_BEAM_WIDTH);
            todo = next_todo;   // reverseしない　->　popはスコアが小さい順に取り出す
        }
        self.ans = todo[0].ans.clone();
        self.score = todo[0].score;
    }

    // 結果出力
    fn result(&self) -> String {
        self.ans.iter().map(|&b| if b { "A" } else { "B" }).join("\n")
    }
}

#[derive(Debug, Clone)]
struct State {
    list: Vec<isize>,
    ans: Vec<bool>,
    score: isize,
    score_diff: isize,
}

impl State {
    fn new(list: Vec<isize>, ans: Vec<bool>, score: isize, score_diff: isize) -> Self {
        Self { list, ans, score, score_diff }
    }

    fn neighbors(&self, e: &Env, t: usize) -> Vec<Self> {
        let mut res = Vec::new();
        let score_diff_org = self.score_diff - e.pqr[t]
            .iter().filter(|&&x| self.list[x] == 0).count() as isize;
        for d in [1, -1] {
            let mut list = self.list.clone();
            e.pqr[t].iter().for_each(|&x| list[x] += d);
            let score_diff = score_diff_org + e.pqr[t]
                .iter().filter(|&&x| list[x] == 0).count() as isize;
            let score = self.score + score_diff;
            let mut ans = self.ans.clone();
            ans.push(d == 1);
            res.push(State::new(list, ans, score, score_diff));
        }
        res
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new(vec![0; N], vec![], 0, N as isize)
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
