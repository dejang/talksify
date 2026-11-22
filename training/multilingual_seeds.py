"""Built-in original seed pairs spanning all 24 non-English Parakeet TDT 0.6B v3 languages.

Every seed uses original domains, disfluency structures, and phrasing (never derived from
held-out recordings or ``cleanup_quality_eval/cases.rs``).
"""

from __future__ import annotations

from dataclasses import dataclass

from prompts import PARAKEET_V3_LANGUAGES
from qa_filters import is_retraction_category


@dataclass(frozen=True)
class MultilingualRow:
    raw: str
    clean: str
    lang: str
    tier: str
    source: str
    category: str
    is_retraction: bool = False


def _seed(
    lang: str,
    category: str,
    raw: str,
    clean: str,
) -> MultilingualRow:
    return MultilingualRow(
        raw=raw,
        clean=clean,
        lang=lang,
        tier=PARAKEET_V3_LANGUAGES[lang].tier,
        source="builtin_multilingual_seed",
        category=category,
        is_retraction=is_retraction_category(category, "builtin_multilingual_seed"),
    )


BUILTIN_MULTILINGUAL_SEEDS: tuple[MultilingualRow, ...] = (
    # --- Tier 1: de, fr, es, it, ro ---
    _seed("de", "scoped_self_correction",
          "Äh, wir haben den Projektor im Raum drei, nein, sorry, im Raum sieben aufgebaut.",
          "Wir haben den Projektor im Raum sieben aufgebaut.",),
    _seed("de", "vad_pause_break",
          "Die Audiodatei wird zuerst komprimiert und. Dann im lokalen Archiv gespeichert.",
          "Die Audiodatei wird zuerst komprimiert und dann im lokalen Archiv gespeichert."),
    _seed("de", "clean_passthrough",
          "Welche Mikrofonkapsel eignet sich am besten für ruhige Aufnahmen im Homeoffice?",
          "Welche Mikrofonkapsel eignet sich am besten für ruhige Aufnahmen im Homeoffice?"),
    _seed("de", "cutoff_false_start",
          "Ähm, wir müssen die Konf die Konfiguration vor dem nächsten Neustart prüfen.",
          "Wir müssen die Konfiguration vor dem nächsten Neustart prüfen."),
    _seed("fr", "scoped_self_correction",
          "Euh, j'ai réservé la salle bleue au deuxième étage, enfin non, au quatrième étage.",
          "J'ai réservé la salle bleue au quatrième étage.",),
    _seed("fr", "vad_pause_break",
          "L'application synchronise le calendrier dès que. L'utilisateur ouvre la fenêtre principale.",
          "L'application synchronise le calendrier dès que l'utilisateur ouvre la fenêtre principale."),
    _seed("fr", "clean_passthrough",
          "Peux-tu exporter le tableau des dépenses en format CSV avant la pause déjeuner ?",
          "Peux-tu exporter le tableau des dépenses en format CSV avant la pause déjeuner ?"),
    _seed("fr", "cutoff_false_start",
          "Euh, il faut simplifier l'arch l'architecture du module d'importation audio.",
          "Il faut simplifier l'architecture du module d'importation audio."),
    _seed("es", "scoped_self_correction",
          "Eh, vamos a guardar las capturas en la carpeta verde, no, perdón, en la carpeta azul.",
          "Vamos a guardar las capturas en la carpeta azul.",),
    _seed("es", "vad_pause_break",
          "El filtro reduce el ruido ambiental sin. Cortar la voz del hablante.",
          "El filtro reduce el ruido ambiental sin cortar la voz del hablante."),
    _seed("es", "clean_passthrough",
          "¿Cuántas licencias de escritorio necesitamos renovar para el equipo de diseño?",
          "¿Cuántas licencias de escritorio necesitamos renovar para el equipo de diseño?"),
    _seed("es", "cutoff_false_start",
          "Ehm, conviene revisar la conf la configuración del micrófono antes de grabar.",
          "Conviene revisar la configuración del micrófono antes de grabar."),
    _seed("it", "scoped_self_correction",
          "Ehm, per il laboratorio abbiamo ordinato quattro monitor, anzi no, sei monitor nuovi.",
          "Per il laboratorio abbiamo ordinato sei monitor nuovi.",),
    _seed("it", "vad_pause_break",
          "La scorciatoia da tastiera resta attiva finché. La finestra rimane aperta sullo schermo.",
          "La scorciatoia da tastiera resta attiva finché la finestra rimane aperta sullo schermo."),
    _seed("it", "clean_passthrough",
          "Puoi controllare se il pacchetto grafico si installa correttamente su Linux?",
          "Puoi controllare se il pacchetto grafico si installa correttamente su Linux?"),
    _seed("it", "cutoff_false_start",
          "Ehm, voglio verificare la conf la configurazione della tastiera sul nuovo portatile.",
          "Voglio verificare la configurazione della tastiera sul nuovo portatile."),
    _seed("ro", "scoped_self_correction",
          "Ăă, am rezervat sala de conferințe la etajul doi, nu, scuze, la etajul patru.",
          "Am rezervat sala de conferințe la etajul patru.",),
    _seed("ro", "scoped_self_correction",
          "Păi, am trimis pachetul către depozitul nordic, stai, nu, către atelierul central.",
          "Am trimis pachetul către atelierul central.",),
    _seed("ro", "vad_pause_break",
          "Programul salvează automat notițele imediat ce. Închidem fereastra de dictare.",
          "Programul salvează automat notițele imediat ce închidem fereastra de dictare."),
    _seed("ro", "clean_passthrough",
          "Câte microfoane USB avem disponibile în studio pentru înregistrarea podcastului?",
          "Câte microfoane USB avem disponibile în studio pentru înregistrarea podcastului?"),
    _seed("ro", "cutoff_false_start",
          "Ăă, hai să verificăm arh arhitectura modulului audio înainte să publicăm.",
          "Hai să verificăm arhitectura modulului audio înainte să publicăm."),
    # --- Tier 2: all 19 remaining Parakeet TDT 0.6B v3 languages ---
    _seed("bg", "scoped_self_correction",
          "Ъъ, сложих кабелите в синята кутия, не, извинявай, в сивата раница.",
          "Сложих кабелите в сивата раница.",),
    _seed("hr", "vad_pause_break",
          "Eee, aplikacija sprema snimku na disk čim. Korisnik otpusti tipku za snimanje.",
          "Aplikacija sprema snimku na disk čim korisnik otpusti tipku za snimanje."),
    _seed("cs", "cutoff_false_start",
          "Ehm, musíme zkontrolovat nast nastavení mikrofonu před začátkem rozhovoru.",
          "Musíme zkontrolovat nastavení mikrofonu před začátkem rozhovoru."),
    _seed("da", "scoped_self_correction",
          "Øhm, vi købte kaffe fra bageriet på hjørnet, nej, undskyld, fra risteriet ved havnen.",
          "Vi købte kaffe fra risteriet ved havnen.",),
    _seed("nl", "vad_pause_break",
          "Uhm, het programma verwerkt de spraak lokaal zodat. Niets naar een externe server gaat.",
          "Het programma verwerkt de spraak lokaal zodat niets naar een externe server gaat."),
    _seed("et", "cutoff_false_start",
          "Eee, peaksime üle vaatama heli helikaardi seaded enne uue klipi salvestamist.",
          "Peaksime üle vaatama helikaardi seaded enne uue klipi salvestamist."),
    _seed("fi", "scoped_self_correction",
          "Öhm, tilasin työpöydälle kaksi lamppua, ei, anteeksi, kolme lamppua.",
          "Tilasin työpöydälle kolme lamppua.",),
    _seed("el", "cutoff_false_start",
          "Εμ, θέλω να ελέγξω την ακουσ ακουστική του δωματίου πριν αρχίσουμε την ηχογράφηση.",
          "Θέλω να ελέγξω την ακουστική του δωματίου πριν αρχίσουμε την ηχογράφηση."),
    _seed("hu", "scoped_self_correction",
          "Öö, a könyvet a konyhaasztalon hagytam, nem, bocs, az íróasztal fiókjában.",
          "A könyvet az íróasztal fiókjában hagytam.",),
    _seed("lv", "vad_pause_break",
          "Ēē, sistēma automātiski saglabā melnrakstu ikreiz kad. Mainām aktīvo logu.",
          "Sistēma automātiski saglabā melnrakstu ikreiz kad mainām aktīvo logu."),
    _seed("lt", "scoped_self_correction",
          "Ehm, išsiunčiau brėžinius į Kauno skyrių, ne, atsiprašau, į Klaipėdos skyrių.",
          "Išsiunčiau brėžinius į Klaipėdos skyrių.",),
    _seed("mt", "scoped_self_correction",
          "Emm, ħallejt iċ-ċwievet fil-kċina, le, skużani, ħdejn il-bieb ta' barra.",
          "Ħallejt iċ-ċwievet ħdejn il-bieb ta' barra.",),
    _seed("pl", "cutoff_false_start",
          "Yyy, warto wyczyścić filtr powietrza przed włącz włączeniem projektora w sali.",
          "Warto wyczyścić filtr powietrza przed włączeniem projektora w sali."),
    _seed("pt", "scoped_self_correction",
          "Hã, comprei bilhetes para o comboio das nove, não, desculpa, das onze da manhã.",
          "Comprei bilhetes para o comboio das onze da manhã.",),
    _seed("ru", "vad_pause_break",
          "Ээ, программа очищает лишние шумы до того как. Текст появляется в редакторе.",
          "Программа очищает лишние шумы до того как текст появляется в редакторе."),
    _seed("sk", "scoped_self_correction",
          "Ehm, odložil som kľúče do hornej zásuvky, nie, prepáč, do spodnej skrinky.",
          "Odložil som kľúče do spodnej skrinky.",),
    _seed("sl", "cutoff_false_start",
          "Emm, pred oddajo moramo preveriti prev prevod menija na začetnem zaslonu.",
          "Pred oddajo moramo preveriti prevod menija na začetnem zaslonu."),
    _seed("sv", "cutoff_false_start",
          "Öhm, vi behöver uppdatera drivrut drivrutinen för ljudkortet på testdatorn.",
          "Vi behöver uppdatera drivrutinen för ljudkortet på testdatorn."),
    _seed("uk", "clean_passthrough",
          "Скільки вільного місця залишилося на зовнішньому диску для резервних копій?",
          "Скільки вільного місця залишилося на зовнішньому диску для резервних копій?"),
)
