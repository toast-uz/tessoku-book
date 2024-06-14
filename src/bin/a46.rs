use std::time::Instant;
use proconio::input;
use itertools::{iproduct, Itertools};
//use rustc_hash::{FxHashSet as HashSet, FxHashMap as HashMap};
//use rust_snippets::xorshift_rand::*;
//use rust_snippets::kyopro_args::*;
use xorshift_rand::*;
use kyopro_args::*;

const LIMIT: f64 = 0.99;
const DEBUG: bool = true;
const SA_PATIENCE: usize = 500;

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
    let mut a = Agent::new(&e);
    a.optimize(&e, &mut rng, &timer, LIMIT);
    println!("{}", a.result(&e));
    dbg!("counter:{}", a.counter);
    dbg!("Computed_score = {}", (1e6 / a.score).floor() as isize);
}

#[derive(Debug, Clone, Default)]
struct Env {
    n: usize,
    dists: Vec<Vec<f64>>,
    shorters: Vec<Vec<Vec<usize>>>,
    sa_patience: usize,
}

impl Env {
    fn new() -> Self {
        input! {
            n: usize,
            mut xy: [(isize, isize); n],
        }
        let mut e = Self::default();
        e.init(n, &xy); e
    }

    // テストが作りやすいように、newとinitを分離
    fn init(&mut self, n: usize, xy: &[(isize, isize)]) {
        // 問題入力の設定
        let n = n + 1;
        let mut xy = xy.to_vec();
        xy.push(xy[0]);
        self.n = n;
        self.dists = vec![vec![0.0; n]; n];
        for (i, j) in iproduct!(0..n, 0..n) {
            self.dists[i][j] = (((xy[i].0 - xy[j].0).pow(2) + (xy[i].1 - xy[j].1).pow(2)) as f64).sqrt();
        }
        self.shorters = vec![vec![Vec::new(); n]; n];
        for i in 0..n {
            let seq = self.dists[i].iter().enumerate()
                .filter(|&(j, _)| i != j)
                .sorted_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(j, _)| j).collect_vec();
            let mut got = Vec::new();
            let mut candidate = Vec::new();
            for &j in &seq {
                if let Some(&last) = candidate.last() {
                    if self.dists[i][last] < self.dists[i][j] {
                        got.extend(&candidate);
                        candidate.clear();
                    }
                }
                self.shorters[i][j] = got.clone();
                candidate.push(j);
            }
        }
        // ハイパーパラメータの設定
        self.sa_patience = os_env::get("sa_patience").unwrap_or(SA_PATIENCE);
    }
}

#[derive(Debug, Clone, Default)]
struct Agent {
    seq: Vec<usize>,
    inv_seq: Vec<usize>,
    score: f64,
    counter: usize,
}

impl Agent {
    fn new(e: &Env) -> Self {
        let mut a = Self::default();
        a.init(e); a
    }

    fn init(&mut self, e: &Env) {
        self.seq = (0..e.n).collect();
        self.inv_seq = (0..e.n).collect();
        self.score = self.compute_score(e);
    }

    fn optimize(&mut self, e: &Env, rng: &mut XorshiftRng, timer: &Instant, limit: f64) {
        let mut best = self.clone();
        while timer.elapsed().as_secs_f64() < limit {
            self.counter += 1;
            // PATIENCE回、ベスト更新されなかったら，現在のカウンターをベストにコピーして、ベストから再開する
            let neighbor = if self.counter > best.counter + e.sa_patience {
                best.counter = self.counter;
                *self = best.clone();
                dbg!("counter:{} score:{:.0} restart from the best and kick", self.counter, self.score);
                self.select_neighbor(e, rng, true)
            } else {
                // 遷移候補を決めて、遷移した場合のコスト差分を計算する
                self.select_neighbor(e, rng, false)
            };
            let score_diff = self.compute_score_diff(e, neighbor);
            // スコアが高いほど良い場合
            // スコアが低いほど良い場合はprob < rng.gen()とする
            if score_diff < 0.0 || neighbor.forced() { // 確率prob or 強制近傍か で遷移する
                self.transfer_neighbor(e, neighbor);
                self.score += score_diff;
                // スコアが高いほど良い場合
                // スコアが低いほど良い場合は self.score < best.score とする
                if best.score > self.score {
                    best = self.clone();
                    dbg!("counter:{} score:{:.0} new best", best.counter, best.score);
                }
            }
        }
        // 現在のベストを最終結果として採用する
        best.counter = self.counter;
        *self = best;
    }

    // 近傍を選択する
    fn select_neighbor(&self, e: &Env, rng: &mut XorshiftRng, kick: bool) -> Neighbor {
        if !kick {
            let mut candidate = Vec::new();
            let mut i_list = (0..(e.n - 2)).collect_vec();
            i_list.shuffle(rng);
            for &i in &i_list {
                // 2-opt が有効かどうかを高速に判定する
                // https://future-architect.github.io/articles/20211201a/#2-opt-ILS
                let (v1, v2) = (self.seq[i], self.seq[i + 1]);
                for &u in &e.shorters[v1][v2] {
                    let j = self.inv_seq[u];
                    if j < i + 1 || j == e.n - 1 { continue; }
                    let neighbor = Neighbor::TwoOpt(i, j);
                    let score_diff = self.compute_score_diff(e, neighbor);
                    if score_diff < 0.0 { candidate.push((score_diff, neighbor)); }
                }
                for &u in &e.shorters[v2][v1] {
                    let j = self.inv_seq[u];
                    if j > i || j == 0 { continue; }
                    let neighbor = Neighbor::TwoOpt(j - 1, i);
                    let score_diff = self.compute_score_diff(e, neighbor);
                    if score_diff < 0.0 { candidate.push((score_diff, neighbor)); }
                }
                if !candidate.is_empty() {
                    return candidate.iter().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap().1;
                }
            }
        }
        // kick
        let v = rng.gen_range_multiple(1..e.n, 4);
        Neighbor::DoubleBridge(v[0], v[1], v[2], v[3])
    }

    // 指定された近傍に遷移する
    fn transfer_neighbor(&mut self, _e: &Env, neighbor: Neighbor) {
        // 近傍遷移
        match neighbor {
            Neighbor::TwoOpt(i, j) => {
                for k in (i + 1)..=((i + j) / 2) {
                    self.seq.swap(k, i + j - k + 1);
                    self.inv_seq[self.seq[k]] = k;
                    self.inv_seq[self.seq[i + j - k + 1]] = i + j - k + 1;
                }
            },
            Neighbor::DoubleBridge(p, q, r, s) => {
                let mut res = Vec::new();
                for i in 0..p { res.push(self.seq[i].clone()); }
                for i in r..s { res.push(self.seq[i].clone()); }
                for i in q..r { res.push(self.seq[i].clone()); }
                for i in p..q { res.push(self.seq[i].clone()); }
                for i in s..self.seq.len() { res.push(self.seq[i].clone()); }
                self.seq = res;
                self.seq.iter().enumerate().for_each(|(i, &x)| self.inv_seq[x] = i);
            },
        }
    }

    // スコアの差分計算
    // 以下のいずれかで実装する
    // 1) cloneしたnew_stateを遷移させて、フル計算する
    // 2) selfを遷移させて、フル計算し、その後、selfを逆遷移させる
    // 3) 差分計算をする
    // 3)の場合は、1)のコードを最初は残して、結果を照合する
    fn compute_score_diff(&self, e: &Env, neighbor: Neighbor) -> f64 {
        match neighbor {
            Neighbor::TwoOpt(i, j) => {
                e.dists[self.seq[i]][self.seq[j]] + e.dists[self.seq[i + 1]][self.seq[j + 1]]
                - e.dists[self.seq[i]][self.seq[i + 1]] - e.dists[self.seq[j]][self.seq[j + 1]]
            },
            Neighbor::DoubleBridge(_, _, _, _) => {
                let score_old = self.score;
                let mut new_state = self.clone();
                new_state.transfer_neighbor(e, neighbor);
                let score_new = new_state.compute_score(e);
                score_new - score_old
            },
        }
    }

    // スコアのフル計算
    fn compute_score(&self, e: &Env) -> f64 {
        self.seq.iter().tuple_windows().map(|(&i, &j)| e.dists[i][j]).sum()
    }

    // 結果出力
    fn result(&self, e: &Env) -> String {
        self.seq.iter().map(|&i| if i < e.n - 1 {i + 1} else { 1 }).join("\n")
    }
}

// 近傍識別
#[derive(Debug, Clone, Copy)]
enum Neighbor {
    TwoOpt(usize, usize), // aとbを交換
    DoubleBridge(usize, usize, usize, usize),
}

impl Neighbor {
    // 近傍を逆遷移させるための近傍を返す
    // kick系の非可逆なNeighborはNone（戻さない）とする
    /*
    fn reversed(&self) -> Self {
        match *self {
            Self::Swap(a, b) => Self::Swap(b, a),
            Self::None => Self::None,
        }
    }
    */
    // 強制で遷移する近傍かどうか
    // kick系の非可逆なNeighborはtrueとする
    #[inline]
    fn forced(&self) -> bool {
        match *self {
            Self::TwoOpt(_, _) => false,
            Self::DoubleBridge(_, _, _, _) => true,
        }
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

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn basic() {
        let n = 5;
        let xy = vec![(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)];
        let mut e = Env::default();
        e.init(n, &xy);
        let mut a = Agent::new(&e);
        eprintln!("{:?}", a);
        a.transfer_neighbor(&e, Neighbor::DoubleBridge(0, 1, 2, 5));
        eprintln!("{:?}", a);
        assert!(false);
    }
}
