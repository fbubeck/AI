?&$	??Q?X&??a5???%???e??E??!?a?? ???$	??V_?@?J??h8?????m?@!H??q?C@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0D?Ö?????t???ABҧU????Y+hZbe4??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0v?1<?3???? ????A?Ȓ9?w??Y??(?????rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?a?? ???	8?*5{??A ??a??Ys?9>Z???rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??S^???߾?3??A-&6׆??Y?%!????rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???W?????Ҩ????Ab?G??Y??*????rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??????@?0`?U??A??V??,??Y?Z??m??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?_????0?????A=???????YB??v?$??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	kF?????Tl????A&??'d???Y?)??z???rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
Qf?L2r???}??ŉ??A?
Ĳ???Y?/??\??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??ne????t??%??A?????0??Y?+,????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??ZH???a?'֩??A?Sb.??YGˁjې?rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????_????F?????A??5???Y??
???rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0@??wԘ??`Z?'????A?6ǹM???YS?h?w??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?WV?????????&??A?3??E`??YSYvQ???rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???n???????a???A??D.8???Y?%?L1??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?e??E??1?JZ???A?N$?jf??Y?j?????rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??g?ej??@?t?_???A{g?UId??YL?
F%u??rtrain 97*	??x?&?~@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?k]j?~??!]?+???A@)???QF\??1??#?r$@@:Preprocessing2T
Iterator::Root::ParallelMapV2??)Wx???!ÛF?2@)??)Wx???1ÛF?2@:Preprocessing2E
Iterator::Rooty=???!?fT3?TB@)??Qٰ???1
a`?1@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???e???!?i>U$@)???e???1?i>U$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?V?????!z???,?O@)??	????1??C?C
#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate.??T???!??U{d-@)??r??ږ?1???y@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapܺ??:???!?i]x*&2@)?????k??1???jf?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?L????!W???=@)?L????1W???=@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 44.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?j??l@I??0џ$X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	)??s???q?Я?@??????&??!	8?*5{??	!       "	!       *	!       2$	k?i????&?{@????=???????!?
Ĳ???:	!       B	!       J$	??d??ϒ?`J??g??j?????!?)??z???R	!       Z$	??d??ϒ?`J??g??j?????!?)??z???b	!       JCPU_ONLYY?j??l@b q??0џ$X@Y      Y@q8? ??uU@"?	
both?Your program is POTENTIALLY input-bound because 44.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?85.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 