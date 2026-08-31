set(ESPEAK_SOURCES
    "src/common.c"
    "src/mnemonics.c"
    "src/error.c"
    "src/ieee80.c"
    "src/compiledata.c"
    "src/compiledict.c"
    "src/dictionary.c"
    "src/encoding.c"
    "src/intonation.c"
    "src/langopts.c"
    "src/numbers.c"
    "src/phoneme.c"
    "src/phonemelist.c"
    "src/readclause.c"
    "src/setlengths.c"
    "src/soundicon.c"
    "src/spect.c"
    "src/ssml.c"
    "src/synthdata.c"
    "src/synthesize.c"
    "src/tr_languages.c"
    "src/translate.c"
    "src/translateword.c"
    "src/voices.c"
    "src/wavegen.c"
    "src/speech.c"
    "src/espeak_api.c"
    "ucd/case.c"
    "ucd/categories.c"
    "ucd/ctype.c"
    "ucd/proplist.c"
    "ucd/scripts.c"
    "ucd/tostring.c"
)

idf_component_register(
    SRCS ${ESPEAK_SOURCES}
    INCLUDE_DIRS "include"
    PRIV_INCLUDE_DIRS "src"
)

target_compile_definitions(
    ${COMPONENT_LIB}
    PRIVATE
        LIBESPEAK_NG_EXPORT=1
        PATH_ESPEAK_DATA="/g2p/espeak-ng-data"
)
target_compile_options(
    ${COMPONENT_LIB}
    PRIVATE
        -fwrapv
        -Wno-error
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-missing-field-initializers
)

