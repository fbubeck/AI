?($	??%	?J??~?N???/n????!?ŏ1w??$	???)@???%@??Tx*<@!???)]?8@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?~j?t???H?z?G??AE???JY??YjM??S??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=yX?5??r??????A??ʡE??Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-?????e?`TR'??A??y?):??Y$????ۗ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??{??P???ׁsF???AM?O????YH?}8g??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>?٬?\???U??????A?8EGr???Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ꕲq?????T?????A?	?c??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?D???J?????????Aŏ1w-!??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?w??#?????:M??A? ?	???Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???<,?????A?f??Aj?q?????Y䃞ͪϕ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?E????????<,Ԛ??A?Y??ڊ??Y0*??D??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?7??d??????V?/??A?g??s???Y???3???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ŏ1w??!?lV}??A	?^)???Y??|?5^??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"?uq???>W[????Aio???T??Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???ZӼ????N@a??A??o_??YaTR'????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}гY????-??????A??ܵ?|??Y]m???{??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&i o??????e??a??A?Fx$??YT㥛? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e????n?????A??????Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???ZӼ???镲q??AM?O???Y2??%䃎?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&\???(\??lxz?,C??A???K7???Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?:pΈ??!?lV}??Aw??/???Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&/n????&S??:??A?uq???Y?N@aÓ?*	33333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat=?U????!o&:?qB@)??	h"l??1??d1?aA@:Preprocessing2F
Iterator::Model??W?2???!e2#֦?B@)?lV}????1?g?)??6@:Preprocessing2U
Iterator::Model::ParallelMapV2?X????!?am?,@)?X????1?am?,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip!?lV}??!???)YyO@)8gDio???1"{??M?"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???QI???!?`;?-H@)???QI???1?`;?-H@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??e?c]??!_qD^$(@)9??m4???1)??ˎ @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???S????!ǐ???0@)??ܥ?1]>???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*+??????!??Y??? @)+??????1??Y??? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??^?.?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	!X?}???? 1????&S??:??!!?lV}??	!       "	!       *	!       2$	Fk?Fk?????|?僷??uq???!	?^)???:	!       B	!       J$	q'p'??l??}e????H?}??!H?}8g??R	!       Z$	q'p'??l??}e????H?}??!H?}8g??JCPU_ONLYY??^?.?@b Y      Y@q؛7???@"?
both?Your program is POTENTIALLY input-bound because 51.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?31.8554% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 