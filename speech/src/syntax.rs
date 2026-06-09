use serde::{Deserialize, Serialize};

use crate::segment::TerminalPunctuation;

pub type WordIndex = usize;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SentenceSyntaxAnalysis {
    pub tokens: Vec<SyntaxToken>,
    pub link_parses: Vec<SyntacticLinkParse>,
    pub terminal: Option<TerminalPunctuation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntaxToken {
    pub word_index: WordIndex,
    pub text: String,
    pub pos: PartOfSpeech,
    pub prosodic_role: ProsodicRole,
    pub syntactic_links: Vec<SyntacticLinkKind>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntacticLinkParse {
    pub links: Vec<SyntacticLink>,
    pub rank: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntacticLink {
    pub left: WordIndex,
    pub right: WordIndex,
    pub kind: SyntacticLinkKind,
    pub confidence: f32,
    pub source: SyntacticLinkSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SyntacticLinkKind {
    Subject,
    Object,
    Complement,
    InfinitivalMarker,
    Modifier,
    Determiner,
    Auxiliary,
    Preposition,
    Coordination,
    ContrastPair,
    NounCompound,
    Vocative,
    Apposition,
    Parenthetical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntacticLinkSource {
    HeuristicGrammarIsland,
    AmbiguityVariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartOfSpeech {
    Noun,
    Verb,
    Auxiliary,
    Determiner,
    Preposition,
    Pronoun,
    Adverb,
    Adjective,
    Conjunction,
    Particle,
    ProperName,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProsodicRole {
    Content,
    FunctionWeak,
    FunctionStrong,
    Contrastive,
    Focus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentPattern {
    pub predicates: Vec<ContextPredicate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextPredicate {
    SyntacticLink(SyntacticLinkKind),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SyntaxRuleContext {
    pub word_links: Vec<WordSyntacticLinks>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordSyntacticLinks {
    pub word_index: WordIndex,
    pub links: Vec<SyntacticLinkKind>,
}

pub trait LinkGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicLinkGrammarParser;

impl LinkGrammarParser for HeuristicLinkGrammarParser {
    fn parse(
        &self,
        words: &[String],
        terminal: Option<TerminalPunctuation>,
    ) -> SentenceSyntaxAnalysis {
        parse_english_link_grammar(words, terminal)
    }
}

pub fn parse_english_link_grammar(
    words: &[String],
    terminal: Option<TerminalPunctuation>,
) -> SentenceSyntaxAnalysis {
    let links = build_links(words);
    let parse = SyntacticLinkParse { links, rank: 1.0 };
    let tokens = words
        .iter()
        .enumerate()
        .map(|(word_index, word)| {
            let mut syntactic_links = parse
                .links
                .iter()
                .filter_map(|link| {
                    (link.left == word_index || link.right == word_index).then_some(link.kind)
                })
                .collect::<Vec<_>>();
            syntactic_links.sort_unstable_by_key(|kind| *kind as u8);
            syntactic_links.dedup();
            SyntaxToken {
                word_index,
                text: word.clone(),
                pos: disambiguate_pos_from_links(word_index, base_pos(word), &parse.links),
                prosodic_role: prosodic_role_for_word(word, &syntactic_links),
                syntactic_links,
            }
        })
        .collect();

    SentenceSyntaxAnalysis {
        tokens,
        link_parses: vec![parse],
        terminal,
    }
}

impl SentenceSyntaxAnalysis {
    pub fn primary_parse(&self) -> Option<&SyntacticLinkParse> {
        self.link_parses.first()
    }

    pub fn environment_patterns(&self) -> Vec<EnvironmentPattern> {
        self.link_parses
            .iter()
            .map(SyntacticLinkParse::as_environment_pattern)
            .collect()
    }

    pub fn rule_context(&self) -> SyntaxRuleContext {
        SyntaxRuleContext {
            word_links: self
                .tokens
                .iter()
                .map(|token| WordSyntacticLinks {
                    word_index: token.word_index,
                    links: token.syntactic_links.clone(),
                })
                .collect(),
        }
    }

    pub fn word_has_link(&self, word_index: WordIndex, kind: SyntacticLinkKind) -> bool {
        self.rule_context().word_has_link(word_index, kind)
    }

    pub fn matches_environment_pattern(&self, pattern: &EnvironmentPattern) -> bool {
        let Some(primary) = self.primary_parse() else {
            return false;
        };
        pattern.predicates.iter().all(|predicate| match predicate {
            ContextPredicate::SyntacticLink(kind) => {
                primary.links.iter().any(|link| link.kind == *kind)
            }
        })
    }
}

impl SyntacticLinkParse {
    pub fn as_environment_pattern(&self) -> EnvironmentPattern {
        let mut seen = std::collections::HashSet::new();
        let predicates = self
            .links
            .iter()
            .filter_map(|link| {
                seen.insert(link.kind)
                    .then_some(ContextPredicate::SyntacticLink(link.kind))
            })
            .collect();
        EnvironmentPattern { predicates }
    }
}

impl SyntaxRuleContext {
    pub fn word_has_link(&self, word_index: WordIndex, kind: SyntacticLinkKind) -> bool {
        self.word_links
            .iter()
            .find(|word| word.word_index == word_index)
            .is_some_and(|word| word.links.contains(&kind))
    }
}

fn build_links(words: &[String]) -> Vec<SyntacticLink> {
    let mut links = Vec::new();
    for (index, window) in words.windows(2).enumerate() {
        let left = window[0].as_str();
        let right = window[1].as_str();
        if left == "to" && is_likely_verb(right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::InfinitivalMarker, 0.92),
            );
        }
        if is_determiner(left) && is_likely_nominal(right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Determiner, 0.83),
            );
        }
        if is_auxiliary(left) && is_likely_verb(right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Auxiliary, 0.82),
            );
        }
        if is_preposition(left) && is_likely_nominal(right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Preposition, 0.8),
            );
        }
        if is_modifier_pair(left, right) {
            push_link(
                &mut links,
                link(index, index + 1, SyntacticLinkKind::Modifier, 0.72),
            );
        }
    }

    push_auxiliary_phrase_links(words, &mut links);
    push_core_clause_links(words, &mut links);
    push_coordination_links(words, &mut links);
    push_contrast_links(words, &mut links);
    links
}

fn push_auxiliary_phrase_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for auxiliary_index in 0..words.len() {
        if !is_auxiliary(&words[auxiliary_index]) {
            continue;
        }
        if let Some(verb_index) = words
            .iter()
            .enumerate()
            .skip(auxiliary_index + 1)
            .take(4)
            .find_map(|(index, word)| is_likely_verb(word).then_some(index))
        {
            push_link(
                links,
                link(
                    auxiliary_index,
                    verb_index,
                    SyntacticLinkKind::Auxiliary,
                    0.82,
                ),
            );
        }
    }
}

fn push_core_clause_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for predicate_index in 0..words.len() {
        if !(is_likely_verb(&words[predicate_index]) || is_auxiliary(&words[predicate_index])) {
            continue;
        }
        if let Some(subject_index) = (0..predicate_index)
            .rev()
            .find(|index| is_likely_nominal(&words[*index]) && !is_preposition(&words[*index]))
        {
            push_link(
                links,
                link(
                    subject_index,
                    predicate_index,
                    SyntacticLinkKind::Subject,
                    0.8,
                ),
            );
        }
        if let Some(object_index) = words
            .iter()
            .enumerate()
            .skip(predicate_index + 1)
            .take(5)
            .find_map(|(index, word)| is_likely_nominal(word).then_some(index))
        {
            push_link(
                links,
                link(
                    predicate_index,
                    object_index,
                    SyntacticLinkKind::Object,
                    0.78,
                ),
            );
        }
    }
}

fn push_coordination_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for conjunction_index in 1..words.len().saturating_sub(1) {
        if !is_coordination_conjunction(&words[conjunction_index]) {
            continue;
        }
        push_link(
            links,
            link(
                conjunction_index - 1,
                conjunction_index + 1,
                SyntacticLinkKind::Coordination,
                0.74,
            ),
        );
        push_link(
            links,
            link(
                conjunction_index,
                conjunction_index + 1,
                SyntacticLinkKind::Coordination,
                0.74,
            ),
        );
    }
}

fn push_contrast_links(words: &[String], links: &mut Vec<SyntacticLink>) {
    for (not_index, word) in words.iter().enumerate() {
        if !matches!(word.as_str(), "not" | "n't") {
            continue;
        }
        if let Some(but_index) = words
            .iter()
            .enumerate()
            .skip(not_index + 1)
            .find_map(|(index, word)| (word == "but").then_some(index))
        {
            push_link(
                links,
                link(not_index, but_index, SyntacticLinkKind::ContrastPair, 0.91),
            );
        }
    }
}

fn link(left: usize, right: usize, kind: SyntacticLinkKind, confidence: f32) -> SyntacticLink {
    SyntacticLink {
        left,
        right,
        kind,
        confidence,
        source: SyntacticLinkSource::HeuristicGrammarIsland,
    }
}

fn push_link(links: &mut Vec<SyntacticLink>, link: SyntacticLink) {
    if !links.iter().any(|existing| {
        existing.left == link.left && existing.right == link.right && existing.kind == link.kind
    }) {
        links.push(link);
    }
}

fn disambiguate_pos_from_links(
    word_index: usize,
    base: PartOfSpeech,
    links: &[SyntacticLink],
) -> PartOfSpeech {
    let has_incoming = |kind| {
        links
            .iter()
            .any(|link| link.right == word_index && link.kind == kind)
    };
    match base {
        PartOfSpeech::Noun if has_incoming(SyntacticLinkKind::Auxiliary) => PartOfSpeech::Verb,
        PartOfSpeech::Verb if has_incoming(SyntacticLinkKind::Determiner) => PartOfSpeech::Noun,
        _ => base,
    }
}

fn prosodic_role_for_word(word: &str, links: &[SyntacticLinkKind]) -> ProsodicRole {
    if links.contains(&SyntacticLinkKind::ContrastPair) {
        ProsodicRole::Contrastive
    } else if is_function_word(word) {
        ProsodicRole::FunctionWeak
    } else if links.contains(&SyntacticLinkKind::Object)
        || links.contains(&SyntacticLinkKind::Complement)
    {
        ProsodicRole::Focus
    } else {
        ProsodicRole::Content
    }
}

fn base_pos(word: &str) -> PartOfSpeech {
    if is_auxiliary(word) {
        PartOfSpeech::Auxiliary
    } else if is_determiner(word) {
        PartOfSpeech::Determiner
    } else if is_preposition(word) {
        PartOfSpeech::Preposition
    } else if is_coordination_conjunction(word) {
        PartOfSpeech::Conjunction
    } else if matches!(
        word,
        "i" | "me" | "you" | "he" | "she" | "it" | "we" | "they" | "them"
    ) {
        PartOfSpeech::Pronoun
    } else if is_likely_verb(word) {
        PartOfSpeech::Verb
    } else {
        PartOfSpeech::Noun
    }
}

fn is_function_word(word: &str) -> bool {
    is_auxiliary(word)
        || is_determiner(word)
        || is_preposition(word)
        || is_coordination_conjunction(word)
}

fn is_auxiliary(word: &str) -> bool {
    matches!(
        word,
        "am" | "are"
            | "aren't"
            | "is"
            | "isn't"
            | "was"
            | "wasn't"
            | "were"
            | "weren't"
            | "do"
            | "don't"
            | "does"
            | "doesn't"
            | "did"
            | "didn't"
            | "have"
            | "haven't"
            | "has"
            | "hasn't"
            | "had"
            | "hadn't"
            | "can"
            | "can't"
            | "could"
            | "couldn't"
            | "will"
            | "won't"
            | "would"
            | "wouldn't"
            | "shall"
            | "should"
            | "shouldn't"
            | "may"
            | "might"
            | "must"
            | "ought"
            | "need"
            | "dare"
    )
}

fn is_determiner(word: &str) -> bool {
    matches!(
        word,
        "a" | "an" | "the" | "this" | "that" | "these" | "those" | "my" | "your" | "our"
    )
}

fn is_preposition(word: &str) -> bool {
    matches!(
        word,
        "about" | "after" | "before" | "for" | "from" | "in" | "into" | "of" | "on" | "to" | "with"
    )
}

fn is_coordination_conjunction(word: &str) -> bool {
    matches!(word, "and" | "or" | "but" | "nor")
}

fn is_likely_nominal(word: &str) -> bool {
    !is_function_word(word)
        || matches!(
            word,
            "i" | "me" | "you" | "he" | "she" | "it" | "we" | "they"
        )
}

fn is_likely_verb(word: &str) -> bool {
    matches!(
        word,
        "be" | "come"
            | "coming"
            | "choose"
            | "go"
            | "hear"
            | "inspect"
            | "make"
            | "parse"
            | "see"
            | "want"
            | "wants"
            | "went"
    ) || word.ends_with("ed")
        || word.ends_with("ing")
}

fn is_modifier_pair(left: &str, right: &str) -> bool {
    matches!(
        left,
        "small" | "big" | "good" | "bad" | "new" | "old" | "bright" | "dark"
    ) && is_likely_nominal(right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_auxiliary_and_coordination_links() {
        let words = ["do", "you", "want", "either", "tea", "or", "coffee"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let analysis = parse_english_link_grammar(&words, Some(TerminalPunctuation::Question));

        assert!(analysis.word_has_link(0, SyntacticLinkKind::Auxiliary));
        assert!(analysis.word_has_link(5, SyntacticLinkKind::Coordination));
        assert!(analysis.matches_environment_pattern(&EnvironmentPattern {
            predicates: vec![ContextPredicate::SyntacticLink(
                SyntacticLinkKind::Coordination
            )],
        }));
    }
}
