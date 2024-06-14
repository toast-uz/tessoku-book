use std::time::Instant;
use proconio::input;
use itertools::{iproduct, Itertools};
use rustc_hash::{FxHashSet as HashSet, FxHashMap as HashMap};
//use rust_snippets::xorshift_rand::*;
//use rust_snippets::kyopro_args::*;
use xorshift_rand::*;
use kyopro_args::*;

const N: usize = 50;
const K: usize = 400;
const L: usize = 20;

const LIMIT: f64 = 0.95;
const DEBUG: bool = true;
const SA_START_TEMP: f64 = 1e5;
const SA_END_TEMP: f64 = 1e2;
const SA_PATIENCE: usize = 100;
const SA_TIMER_RESOLUTION: usize = 10;

#[allow(unused_macros)]
macro_rules! dbg {( $( $x:expr ),* ) => ( if DEBUG {eprintln!($( $x ),* );}) }
#[allow(unused_macros)]
macro_rules! dbg2 {( $( $x:expr ),* ) => ( if DEBUG {
    eprintln!($( $x ),* );
    println!("## {}", format!($( $x ),* ));
}) }

fn main() {
    let timer = Instant::now();
    let mut rng = XorshiftRng::from_seed(1e18 as u64);
    let e = Env::new();
    let mut a = Agent::new(&e, &mut rng);
    a.optimize(&e, &mut rng, &timer, LIMIT);
    println!("{}", a.result());
    dbg!("counter:{}", a.counter);
    dbg!("Computed_score = {}", a.score);
}

#[derive(Debug, Clone, Default)]
struct Env {
    a: Vec<usize>,  // 人口
    b: Vec<usize>,  // 役所職員数
    a_mean: usize,    // 人口平均（エリアとして）
    b_mean: usize,    // 役所職員数平均（エリアとして）
    adj: Vec<HashSet<usize>>,  // 隣接状態
    sa_start_temp: f64,
    sa_duration_temp: f64,
    sa_patience: usize,
    sa_timer_resolution: usize,
}

impl Env {
    fn new() -> Self {
        input! {
            n: usize, k: usize, _l: usize,
            ab: [(usize, usize); k],
            c: [[usize; n]; n],
        }
        let mut e = Self::default();
        e.init(ab, c); e
    }

    // テストが作りやすいように、newとinitを分離
    fn init(&mut self, ab: Vec<(usize, usize)>, c: Vec<Vec<usize>>) {
        // 問題入力の設定
        self.a = ab.iter().map(|x| x.0).collect();
        self.b = ab.iter().map(|x| x.1).collect();
        let a_sum = self.a.iter().sum::<usize>();
        let b_sum = self.b.iter().sum::<usize>();
        self.a_mean = a_sum / L;
        self.b_mean = b_sum / L;
        let mut adj: Vec<HashSet<usize>> = vec![HashSet::default(); K];
        for (i, j) in iproduct!(0..N, 0..N) {
            let mut neighbor = Vec::new();
            if i < N - 1 { neighbor.push((i + 1, j)); }
            if j < N - 1 { neighbor.push((i, j + 1)); }
            for &(i2, j2) in &neighbor {
                let x = c[i][j];
                let y = c[i2][j2];
                if x > 0 && y > 0 && x != y {
                    adj[x - 1].insert(y - 1);
                    adj[y - 1].insert(x - 1);
                }
            }
        }
        self.adj = adj;
        // ハイパーパラメータの設定
        self.sa_start_temp = os_env::get("sa_start_temp").unwrap_or(SA_START_TEMP);
        let sa_end_temp = os_env::get("sa_end_temp").unwrap_or(SA_END_TEMP);
        self.sa_duration_temp = sa_end_temp - self.sa_start_temp;
        self.sa_patience = os_env::get("sa_patience").unwrap_or(SA_PATIENCE);
        self.sa_timer_resolution = os_env::get("sa_timer_resolution").unwrap_or(SA_TIMER_RESOLUTION);
    }
}

#[derive(Debug, Clone, Default)]
struct Agent {
    ans: HashMap<usize, Area>,   // id -> エリア情報
    inv: Vec<usize>,             // 地区 -> id
    score: isize,
    counter: usize,
}

impl Agent {
    fn new(e: &Env, rng: &mut XorshiftRng) -> Self {
        let mut a = Self::default();
        a.init(e, rng); a
    }

    fn init(&mut self, e: &Env, rng: &mut XorshiftRng) {
        self.ans.clear();
        self.inv.clear();
        for i in 0..K {
            self.ans.insert(i, Area::new(e, i));
            self.inv.push(i);
        }
        self.climb_greedy(e, rng);
        self.score = self.compute_score(e);
    }

    fn area_num(&self) -> usize { self.ans.len() }

    // 隣接エリアを列挙
    fn adj_areas(&self, e: &Env, i: usize) -> Vec<usize> {
        let mut res = HashSet::default();
        for &j in &self.ans.get(&i).unwrap().c {
            for &k in &e.adj[j] {
                let id = self.inv[k];
                if id != i { res.insert(id); }
            }
        }
        res.iter().cloned().collect()
    }

    // 隣接エリアを併合
    fn unite(&mut self, _e: &Env, i: usize, j: usize) {
        let x = self.ans.get(&i).unwrap().c.iter().cloned().collect_vec();
        let y = self.ans.get(&j).unwrap().c.iter().cloned().collect_vec();
        let c = x.iter().chain(&y).cloned().collect_vec();
        for &k in &y { self.inv[k] = i; }
        let a = self.ans[&i].a + self.ans[&j].a;
        let b = self.ans[&i].b + self.ans[&j].b;
        self.ans.remove(&i);
        self.ans.remove(&j);
        let new_area = Area { id_: i, a, b, c };
        self.ans.insert(i, new_area.clone());
    }

    // 特定エリアを分解
    fn decompose(&mut self, e: &Env, i: usize) {
        let x = self.ans.get(&i).unwrap().c.iter().cloned().collect_vec();
        self.ans.remove(&i);
        for &k in &x {
            let new_area = Area::new(e, k);
            let new_id = new_area.id();
            self.ans.insert(new_id, new_area.clone());
            self.inv[k] = new_id;
        }
    }

    // 山登りによりL個のエリアに集約する（貪欲法）
    // 実際は同じレベルのエリアを区別しないため、乱択効果が出る
    fn climb_greedy(&mut self, e: &Env, _rng: &mut XorshiftRng) {
        while self.area_num() > L {
            // 人口の平均比+役所職員数の平均比 が最も小さいエリアを統合する
            let i = self.ans.keys()
                .min_by_key(|&&id| self.ans[&id].population(e)).cloned().unwrap();
            // 隣接エリアも同様に選択
            let j = self.adj_areas(e, i).iter()
                .min_by_key(|&&id| self.ans[&id].population(e)).cloned().unwrap();
         self.unite(e, i, j);    // 2つのエリアを統合
        }
    }

    // 特定エリアと隣接エリアを破壊
    fn kick(&mut self, e: &Env, rng: &mut XorshiftRng) {
        let i = self.ans.keys().cloned().collect_vec().choose(rng);  // 起点エリアをランダムに選択
        for j in self.adj_areas(e, i).iter().chain(&[i]) {  // 起点+隣接すべて
            self.decompose(e, *j);  // 選択エリアを分解
        }
        self.climb_greedy(e, rng);  // 山登りによりL個のエリアに再度集約する
        self.score = self.compute_score(e);  // スコアを再計算
    }

    fn optimize(&mut self, e: &Env, rng: &mut XorshiftRng, timer: &Instant, limit: f64) {
        let start_time = timer.elapsed().as_secs_f64();
        let mut best = self.clone();
        let mut temp = e.sa_start_temp;
        loop {
            // 現在の温度を計算（少し重いので計算は sa_timer_resolution 間隔で行う）
            if self.counter % e.sa_timer_resolution == 0 {
                let time = timer.elapsed().as_secs_f64();
                if time >= limit { break; }
                temp = e.sa_start_temp + e.sa_duration_temp * (time - start_time) / (limit - start_time);
            }
            self.counter += 1;
            // PATIENCE回、ベスト更新されなかったら，現在のカウンターをベストにコピーして、ベストから再開する
            if self.counter > best.counter + e.sa_patience {
                best.counter = self.counter;
                *self = best.clone();
                dbg!("counter:{} score:{} restart from the best", self.counter, self.score);
            }
            // 遷移候補を決めて、遷移した場合のコスト差分を計算する
            let mut state = self.clone();
            state.kick(e, rng);
            let score_diff = state.score - self.score;
            // スコアが高いほど良い場合
            // スコアが低いほど良い場合はprob < rng.gen()とする
            let prob = (score_diff as f64 / temp).exp();
            if prob > rng.gen() { // 確率prob or 強制近傍か で遷移する
                *self = state;
                // スコアが高いほど良い場合
                // スコアが低いほど良い場合は self.score < best.score とする
                if best.score < self.score {
                    best = self.clone();
                    dbg!("counter:{} score:{} new best", best.counter, best.score);
                }
            }
        }
        // 現在のベストを最終結果として採用する
        best.counter = self.counter;
        *self = best;
    }

    // スコアのフル計算
    fn compute_score(&self, e: &Env) -> isize {
        assert_eq!(self.area_num(), L);
        let mut p: HashMap<usize, usize> = HashMap::default();
        let mut q: HashMap<usize, usize> = HashMap::default();
        for k in 0..K {
            *p.entry(self.inv[k]).or_default() += e.a[k];
            *q.entry(self.inv[k]).or_default() += e.b[k];
        }
        let pmax = p.values().cloned().max().unwrap() as f64;
        let pmin = p.values().cloned().min().unwrap() as f64;
        let qmax = q.values().cloned().max().unwrap() as f64;
        let qmin = q.values().cloned().min().unwrap() as f64;
        (1e6 * (pmin / pmax).min(qmin / qmax)).round() as isize
    }

    // 結果出力
    fn result(&self) -> String {
        assert_eq!(self.area_num(), L);
        let mut res = vec![0; K];
        for (l, v) in self.ans.values().enumerate() {
            for &k in &v.c { res[k] = l + 1; }
        }
        res.iter().join("\n")
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
struct Area {
    id_: usize,     // 代表地区ID
    a: usize,       // 人口
    b: usize,       // 役所職員数
    c: Vec<usize>,  // 所属地区
}

impl Area {
    fn new(e: &Env, id: usize) -> Self {
        Self { id_: id, a: e.a[id], b: e.b[id], c: vec![id] }
    }
    #[inline]
    fn id(&self) -> usize { self.id_ }

    fn population(&self, e: &Env) -> isize {
        // 乱択効果が出るため、人口と役所職員数の倍数は少なくする
        ((5 * self.a / e.a_mean) * (5 * self.b / e.b_mean)) as isize
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

mod kyopro_args {
    #![allow(dead_code)]

    use regex::Regex;
    use itertools::Itertools;

    pub struct Args {
        args_str: String,
    }

    impl Args {
        pub fn new() -> Self {
            let args: Vec<String> = std::env::args().collect();
            let args_str = args[1..].iter().join(" ");
            Self { args_str }
        }

        pub fn get<T: std::str::FromStr>(&self, arg_name: &str) -> Option<T> {
            let re_str = format!(r"-{}=([\d.-]+)", arg_name);
            let re = Regex::new(&re_str).unwrap();
            let Some(captures) = re.captures(&self.args_str) else { return None; };
            captures[1].parse().ok()
        }
    }

    pub mod os_env {
        const PREFIX: &str = "AHC_PARAMS_";

        pub fn get<T: std::str::FromStr>(name: &str) -> Option<T> {
            let name = format!("{}{}", PREFIX, name.to_uppercase());
            std::env::var(name).ok()?.parse().ok()
        }
    }
}
