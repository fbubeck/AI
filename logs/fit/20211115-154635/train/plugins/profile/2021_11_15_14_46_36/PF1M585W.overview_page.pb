?($	L??*|??j?{w??? o?ŏ??!?b?=y??$	n???)@???Ga:@??iqo?@!??????2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$/?$????L?
F%u??A??T?????Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&e?`TR'????	h"l??A?4?8EG??Yݵ?|г??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&HP?s?????K7?A??A?l??????Y?5?;Nё?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&.?!??u???I+???A??ܵ?|??Y2??%䃎?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8gDio???z6?>W[??AK?=?U??Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& A?c?]??n????A??m4????Y?X?? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&S??:??sh??|???AX?5?;N??YD?l?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???ZӼ???ׁsF???A;M?O??Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?G?z????I+???Aŏ1w-!??Y?]K?=??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	J{?/L???.???1???AΪ??V???YW[??재?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?b?=y???\m?????A?[ A???YY?8??m??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d?]K????O??e??A???????YM?J???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?HP???)?Ǻ???A?	h"lx??Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?lV}????ݵ?|г??A?G?z??Yp_?Q??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&*:??H??-???????A?ŏ1w??YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d]?Fx???!??u???A????S??YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&9??v?????ǘ?????A?1??%???Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&U0*????OjM???AL7?A`???Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ŏ1w??s??A??A?٬?\m??Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&m?????????a??4??A?{??Pk??Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& o?ŏ???sF????A?:pΈҾ?Y???&??*	    ؉@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatx??#????!f?????B@)ffffff??1?.??aA@:Preprocessing2F
Iterator::Model<?R?!???!??R???@@)O??e?c??1???44@:Preprocessing2U
Iterator::Model::ParallelMapV2S?!?uq??!?????)@)S?!?uq??1?????)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipr?鷯??!??V8>?P@)?(\?????1 U??F?'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceX9??v??!c3nZ?@)X9??v??1c3nZ?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateB>?٬???!???VE`+@)t$???~??1;???0?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_?L?J??!??ٳ?G1@)???B?i??1D?B??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*a??+e??!-]I	??@)a??+e??1-]I	??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ȏҔ?6@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	{?.??????Sl????sF????!?\m?????	!       "	!       *	!       2$	??\H?R??i???$???:pΈҾ?!?[ A???:	!       B	!       J$	1??<?????:b??????Pk?w??!?v??/??R	!       Z$	1??<?????:b??????Pk?w??!?v??/??JCPU_ONLYYȏҔ?6@b Y      Y@q4?0M?E?@"?
both?Your program is POTENTIALLY input-bound because 53.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?31.2723% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 