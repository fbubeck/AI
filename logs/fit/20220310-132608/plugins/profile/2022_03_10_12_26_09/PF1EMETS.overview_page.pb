?&$	+??y?:??x?m?C??f???8 ??!?͍?	???$	??лrw@??=????ܱ??"@!??58]?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0σ??v[??|?????AQ??????Yj??&kԓ?rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?P??C??? ֪]??A???u???Y+??????rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0k??P????y=????A5'/2???YG??R^+??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?H?F?q???[ A???A???	????Y?iP4`??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0f???8 ??^M??????A??8G??Y|&??i???rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0(??9x&??Ox	N} ??Aan?r???Y???);???rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0d?g^?????t????A??.\s??Y??^??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	K\Ǹ?b??.??H???AP?>????Y^?c@?z??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?	?s???? [????A4g}?1Y??Y?????Q??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??n??????c?????A?7?q????Y`??i???rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?^)K????????A?????YĲ?CR??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?͍?	???N??????A3p@KW???Yh>?n?K??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?N??????2??(??Ag??I}Y??Y?'i???rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0G?tF^???&3?Vz??A??m?2??YJ??4*p??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Ͽ]?????DL?$z??Aޏ?/????Y?7j??{??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????G???fHū??A?@?ش??Y?Ϲ????rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0F?2????L???<??A??"??~??Y4??ؙB??rtrain 97*	??S??j?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??0?q???!Q|?{??C@)??????1}?E??A@:Preprocessing2T
Iterator::Root::ParallelMapV2i?V?θ?!²?@1/@)i?V?θ?1²?@1/@:Preprocessing2E
Iterator::Root>Ab?{???!D?G??>@)??D2??1?KY??l.@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??c?? ??!+??}??#@)??c?? ??1+??}??#@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?K?;????!/?.FLQ@)G??ά?1[hL#H"@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??7?ܘ??!?????0@)??????1???~߾@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?R%??R??!?'?q@)?R%??R??1?'?q@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapZ*oG8-??!?S?W4@)g,??N??1?/?d?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no97?h?h@Iޘ???$X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?'??߀??????S???|?????!?2??(??	!       "	!       *	!       2$	?? Fd????#????P?>????!3p@KW???:	!       B	!       J$	?-Qa???|??.??k???^??!?7j??{??R	!       Z$	?-Qa???|??.??k???^??!?7j??{??b	!       JCPU_ONLYY7?h?h@b qޘ???$X@Y      Y@q???\ T@"?	
both?Your program is POTENTIALLY input-bound because 45.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?80.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 