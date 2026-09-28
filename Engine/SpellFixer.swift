import Foundation
#if os(iOS)
import UIKit
#else
import AppKit
#endif

/// The system spellchecker (the one behind the red underlines): the light way
/// to get from what was WRITTEN to what was MEANT, with no AI model. It fixes
/// misspelled words ("freind" -> "friend") but not real words in the wrong
/// place ("their" for "there"); that takes Gemma.
enum SpellFixer {
    static func fix(_ text: String, language: String = "en_US") -> String {
        var out = ""
        var word = ""
        func flush() {
            out += word.isEmpty ? "" : correct(word, language: language)
            word = ""
        }
        for ch in text {
            if ch.isLetter || ch == "'" { word.append(ch) } else { flush(); out.append(ch) }
        }
        flush()
        return out
    }

    private static func correct(_ word: String, language: String) -> String {
        guard word.count > 1, let guess = firstGuess(word, language: language) else { return word }
        // keep the writer's capitals
        if word.first!.isUppercase { return guess.prefix(1).uppercased() + guess.dropFirst() }
        return guess
    }

    private static func firstGuess(_ word: String, language: String) -> String? {
        let range = NSRange(location: 0, length: (word as NSString).length)
        #if os(iOS)
        let checker = UITextChecker()
        let bad = checker.rangeOfMisspelledWord(in: word, range: range, startingAt: 0, wrap: false, language: language)
        guard bad.location != NSNotFound else { return nil }
        return checker.guesses(forWordRange: bad, in: word, language: language)?.first
        #else
        let checker = NSSpellChecker.shared
        let bad = checker.checkSpelling(of: word, startingAt: 0, language: language, wrap: false, inSpellDocumentWithTag: 0, wordCount: nil)
        guard bad.location != NSNotFound else { return nil }
        return checker.guesses(forWordRange: bad, in: word, language: language, inSpellDocumentWithTag: 0)?.first
        #endif
    }
}
